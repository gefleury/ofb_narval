from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from narval.answermanager import AnswerManager
from narval.pagefinder import PageFinder
from narval.pdfreader import PDFReader
from narval.prompts import NO_ANSWER_TAG
from narval.qamodel import QAModel
from narval.tableextractor import TableExtractor
from narval.utils import FileSystem, get_data_dir

FS = FileSystem()
DATA_DIR = get_data_dir()


class Pipeline:
    pdf_path = Path(DATA_DIR + "/data/input/pdfs/")
    question_path = Path(DATA_DIR + "/data/input/")
    indic_path = Path(DATA_DIR + "/data/input/")
    output_path = Path(DATA_DIR + "/data/output/")

    def __init__(
        self,
        benchmark_version: str,
        indicator_file: str,
        question_file=None,
        extract_tables=True,
        only_table_search_in_rad=True,
        text_extraction_method="PyPDF2",
        table_extraction_method="PDFPlumber",
        table_answer_filter=True,
        model_name=None,
        prompt_version=None,
    ):
        """
        Initialize the pipeline with the required parameters.
        """
        # Output path
        self.answer_path = Pipeline.output_path / benchmark_version / "answers/"
        # Input paths
        qpath = Pipeline.question_path
        self.question_file_path = qpath / question_file if question_file else qpath
        self.indicator_file_path = Pipeline.indic_path / indicator_file
        # Indicator dataframe
        self.indicator_df = FS.read_csv_to_df(
            self.indicator_file_path,
            usecols=["code_ip", "unit_tag", "prompt_instruction"],
        )
        self.indicator_df["prompt_instruction"] = self.indicator_df[
            "prompt_instruction"
        ].replace(np.nan, "")
        # Benchmark version
        self.benchmark_version = benchmark_version
        # PDF extraction
        self.extract_tables = extract_tables
        self.text_extraction_method = text_extraction_method
        self.table_extraction_method = table_extraction_method
        # Filtering of table answers
        self.table_answer_filter = table_answer_filter
        # Search answers only in tables for RADs
        self.only_table_search_in_rad = only_table_search_in_rad
        # Model attributes
        self.model_name = model_name
        self.prompt_version = prompt_version
        self.qa_model = None
        self.no_answer_tag = NO_ANSWER_TAG

    def extract_text_and_tables_from_pdf(
        self,
        pdf_file,
    ):
        """
        Extract text and tables from the provided PDF file
        """
        pdf_reader = PDFReader(
            Pipeline.pdf_path / pdf_file,
            extract_tables=self.extract_tables,
            text_extraction_method=self.text_extraction_method,
            table_extraction_method=self.table_extraction_method,
        )
        pdf_pages = pdf_reader.textpages
        pdf_tables = pdf_reader.dftables
        toc_indices = pdf_reader.find_toc_pages()
        is_rad = pdf_reader.is_rad()

        return pdf_pages, pdf_tables, toc_indices, is_rad

    def extract_tables_from_pdf(
        self,
        pdf_file,
    ):
        """
        Extract tables from the provided PDF file
        """
        pdf_reader = PDFReader(
            Pipeline.pdf_path / pdf_file,
            extract_text=False,
            extract_tables=True,
            table_extraction_method=self.table_extraction_method,
        )
        pdf_tables = pdf_reader.dftables

        return pdf_tables

    def get_segmentation_df(self, pdf_pages, pdf_tables, competence, toc_indices):
        """
        Get the segmentation_df containing the indices of relevant PDF pages for each question
        """
        pagefinder = PageFinder(self.question_file_path, competence)
        segmentation_df = pagefinder.get_segmentation_df(
            pdf_pages, pdf_tables, toc_indices
        )

        return segmentation_df

    def extract_indicators_from_tables(self, pdf_tables, segmentation_df, year):
        """
        Extract indicator values from relevant tables in pdf_tables given by segmentation_df

        Returns
        -------
        A dictionnary with indicator codes and extracted indicator values (before cleaning)
        """
        answer_list_list = []
        indic_code_list = []

        sub_segmentation_df = segmentation_df.query("table_relevant_pages.notnull()")

        for idx in sub_segmentation_df.index:
            # Be careful memory is accessed with loc
            indicator, _, _, _, table_relevant_pages_list = sub_segmentation_df.loc[idx]
            # Extract the answer_list for this indicator
            answer_list = []
            for page_num, table_num in table_relevant_pages_list:
                df = pdf_tables[page_num][table_num]
                table = TableExtractor(df, self.question_file_path, year)
                answer = table.extract_indicator_value(indicator)
                answer_list.append(answer)

            indic_code_list.append(indicator)
            answer_list_list.append(answer_list)

        # Update the dict
        indicator_value_dict = {
            "indicator_code_list": indic_code_list,
            "answer_list_from_tables": answer_list_list,
        }

        return indicator_value_dict

    def get_known_indicator_list(self, indicator_value_dict: dict) -> list[str]:
        """
        Returns the list of indicator codes from indicator_value_dict
        whose indicator values are considered valid

        Parameter
        ---------
        indicator_value_dict: output of extract_indicators_from_tables
        """
        indic_code_list = indicator_value_dict["indicator_code_list"]
        answer_list_list = indicator_value_dict["answer_list_from_tables"]

        def is_valid_answer(answer):
            c1 = answer != "Not found"
            if self.table_answer_filter:
                # at least one digit in answer
                c2 = any(char.isdigit() for char in answer)
            else:
                c2 = True
            return c1 and c2

        def clean_answer_list(answer_list):
            answer_list = [answer for answer in answer_list if is_valid_answer(answer)]
            return answer_list

        known_indicator_list = [
            code
            for (idx, code) in enumerate(indic_code_list)
            if clean_answer_list(answer_list_list[idx]) != []
        ]

        return known_indicator_list

    def get_indicator_list(self, competence: str) -> list[str]:
        indicator_list = (
            FS.read_csv_to_df(self.question_file_path, usecols=["indic", "competence"])
            .query("competence==@competence")["indic"]
            .unique()
            .tolist()
        )
        return indicator_list

    def are_all_indicators_extracted_from_tables(
        self, known_indicator_list: list[str], competence: str
    ) -> bool:
        indicator_list = self.get_indicator_list(competence)
        return set(indicator_list) == set(known_indicator_list)

    def load_qa_model(self):
        """
        Lazily initialize the QAModel if it hasn't been initialized yet.
        """
        if self.qa_model is None:
            self.qa_model = QAModel(model_name=self.model_name)

    def format_prompt(
        self, question, context, competence, year, collectivity, indicator
    ):
        """
        Returns the prompt corresponding to the input question, context,
        competence, year, collectivity and indicator
        """
        df = self.indicator_df.query("code_ip==@indicator")
        # Get the indictor unit (used in the prompt)
        unit_tag = df["unit_tag"].values[0]
        # Get specific prompt instructions for the indicator
        specific_instruction = (
            df["prompt_instruction"]
            .values[0]
            .format(
                no_answer_tag=self.no_answer_tag,
                unit_tag=unit_tag,
                year=str(year),
                year_plus_1=str(int(year) + 1),
            )
        )
        # Format the prompt
        prompt_params = {
            "context": context,
            "question": question,
            "unit_tag": unit_tag,
            "specific_instruction": specific_instruction,
            "year": year,
            "competence": competence,
            "collectivity": collectivity,
        }

        prompt = self.qa_model.format_prompt(prompt_params, self.prompt_version)

        return prompt

    def get_default_question_answer_dict(
        self, segmentation_df: pd.DataFrame, known_indicator_list: list[str]
    ) -> dict:
        """
        Returns the dictionary with questions and default answers for indicators in known_indicator_list
        """
        sub_segm_df = segmentation_df.query("indicator in @known_indicator_list")
        question_list = sub_segm_df["question"].values.tolist()
        answer = None  # "Indicator value extracted from tables"
        answer_list_list = [answer for i in range(len(sub_segm_df))]

        question_answer_dict = {
            "question": question_list,
            "answer_list": answer_list_list,
        }

        return question_answer_dict

    def ask_one_question(
        self,
        question,
        relevant_pages_list,
        pdf_pages,
        *prompt_params,
        **kwargs,
    ):
        answer_list = []
        for page_num in relevant_pages_list:
            try:
                context = pdf_pages[page_num]
                prompt = self.format_prompt(question, context, *prompt_params)
                answer = self.qa_model.predict(prompt, **kwargs)
                answer_list.append(answer)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()  # Free up any unused memory
                    print(
                        f"Page {page_num} corresponding to question '{question}' has been ignored to avoid memory error"
                    )
                else:
                    raise e  # Re-raise the error if it's not an out-of-memroy error

        return answer_list

    def ask_questions(
        self,
        pdf_pages,
        segmentation_df,
        known_indicator_list,
        competence,
        year,
        collectivity,
        **kwargs,
    ):
        """
        Use the question-answering model to get answers from the PDF
        for indicators that are not in known_indicator_list

        Returns
        -------
        A dictionnary with questions and raw answers (before answer cleaning)
        """
        answer_list_list = []
        question_list = []

        # Ask the language model for indicators that have not been extracted from tables
        sub_segm_df = segmentation_df.query("indicator not in @known_indicator_list")
        # ... Loop on questions
        for idx in tqdm(sub_segm_df.index):  # is it really faster than iterrows ?
            # Be careful memory is accessed with loc
            indicator, question, _, relevant_pages_list, _ = sub_segm_df.loc[idx]
            # Compute the answer_list for this question with the language model
            prompt_params = (competence, year, collectivity, indicator)
            answer_list = self.ask_one_question(
                question, relevant_pages_list, pdf_pages, *prompt_params, **kwargs
            )
            question_list.append(question)
            answer_list_list.append(answer_list)

        # Update the dict
        question_answer_dict = {
            "question": question_list,
            "answer_list": answer_list_list,
        }

        return question_answer_dict

    def clean_answers_from_dict(
        self,
        segmentation_df,
        question_answer_dict,
        indicator_value_dict,
        year,
        textpages=None,
    ):
        """
        Clean the raw answers obtained from the model and stored in question_answer_dict.
        """
        answer_manager = AnswerManager(self.indicator_file_path)
        answer_manager.build_detailed_answer_df(
            segmentation_df, question_answer_dict, indicator_value_dict
        )
        answer_manager.apply_full_cleaning_pipeline(
            textpages=textpages, forbidden_number_list=[float(year)]
        )

        return answer_manager.answer_df.copy(), answer_manager.detailed_answer_df.copy()

    def clean_answers_from_file(
        self,
        data_to_clean_benchmark_version: str,
        detailed_answer_file: str,
        year: str,
        textpages=None,
    ):
        """
        Clean the raw answers stored in the folder
        "/data/output/" + data_to_clean_benchmark_version + "/answers/" + detailed_answer_file
        """
        path = Pipeline.output_path / data_to_clean_benchmark_version / "answers"
        answer_manager = AnswerManager(self.indicator_file_path)
        answer_manager.get_detailed_answer_df_from_file(path / detailed_answer_file)
        answer_manager.apply_full_cleaning_pipeline(
            textpages=textpages, forbidden_number_list=[float(year)]
        )

        return answer_manager.answer_df.copy(), answer_manager.detailed_answer_df.copy()

    def save_answers(
        self,
        answer_df: pd.DataFrame,
        detailed_answer_df: pd.DataFrame,
        pdf_file: str,
        competence: str,
        year: str,
    ):
        """
        Save the cleaned answers.
        """
        for df, tag in zip(
            [answer_df, detailed_answer_df], ["answers", "detailed_answers"]
        ):
            # Add columns in the dataframe
            df.insert(0, "pdf_name", pdf_file)
            df.insert(0, "competence", competence)
            df.insert(0, "year", year)
            df.insert(0, "benchmark_version", self.benchmark_version)
            # Save answers
            self.answer_path.mkdir(parents=True, exist_ok=True)
            answer_df_file = f"{pdf_file.split(".")[0]}_{tag}.csv"
            answer_path = self.answer_path / answer_df_file
            FS.write_df_to_csv(df, answer_path, index=False)

    def run(
        self,
        pdf_file: str,
        competence: str,
        year: str,
        collectivity: str,
        remove_hallucinations=True,
        **kwargs,
    ):
        """
        Execute the full pipeline.
        """
        print(f"Extract text (and tables) from pdf {pdf_file} ...")
        pdf_pages, pdf_tables, toc_indices, is_rad = (
            self.extract_text_and_tables_from_pdf(pdf_file)
        )
        print("Done")
        print("Get the segmentation dataframe  ...")
        segmentation_df = self.get_segmentation_df(
            pdf_pages, pdf_tables, competence, toc_indices
        )
        print("Done")
        print("Extract indicator values from summary tables ...")
        indicator_value_dict = self.extract_indicators_from_tables(
            pdf_tables, segmentation_df, year
        )
        print("Done")
        known_indicator_list = self.get_known_indicator_list(indicator_value_dict)
        default_question_answer_dict = self.get_default_question_answer_dict(
            segmentation_df, known_indicator_list
        )

        if self.are_all_indicators_extracted_from_tables(
            known_indicator_list, competence
        ):
            print("All indicators are extracted from summary tables")
            question_answer_dict = default_question_answer_dict
        elif is_rad and self.only_table_search_in_rad:
            # Quick and dirty way to treat this case
            # Should rather extract only tables and first page from RADs
            print(
                "PDF detected as a RAD. The process is stopped without asking the QA model."
            )
            question_answer_dict = self.get_default_question_answer_dict(
                segmentation_df, self.get_indicator_list(competence)
            )
        else:
            print("Some indicators could not be extracted from summary tables")
            print(f"Loading QA model {self.model_name} ...")
            self.load_qa_model()
            print("QA model loaded.")
            print(f"Asking questions using model {self.model_name} ...")
            llm_question_answer_dict = self.ask_questions(
                pdf_pages,
                segmentation_df,
                known_indicator_list,
                competence,
                year,
                collectivity,
                **kwargs,
            )
            question_answer_dict = merge_question_answer_dicts(
                llm_question_answer_dict, default_question_answer_dict
            )
            print("Done")

        print("Cleaning answers ...")
        textpages = pdf_pages if remove_hallucinations else None
        answer_df, detailed_answer_df = self.clean_answers_from_dict(
            segmentation_df,
            question_answer_dict,
            indicator_value_dict,
            year,
            textpages=textpages,
        )
        print("Done")
        print(f"Saving answers for {pdf_file} ...")
        self.save_answers(answer_df, detailed_answer_df, pdf_file, competence, year)
        print("Done")

    def run_table_extraction_step(
        self,
        pdf_file: str,
        competence: str,
        year: str,
    ):
        """
        Execute only the step which consists of reading a PDF
        and extracting indicators from tables
        """
        print(f"Extract tables from pdf {pdf_file} ...")
        pdf_tables = self.extract_tables_from_pdf(pdf_file)
        print("Done")
        print("Get the segmentation dataframe  ...")
        segmentation_df = self.get_segmentation_df(
            pdf_pages=[], pdf_tables=pdf_tables, competence=competence, toc_indices=None
        )
        print("Done")
        print("Extract indicator values from summary tables ...")
        indicator_value_dict = self.extract_indicators_from_tables(
            pdf_tables, segmentation_df, year
        )
        print("Done")

        print("Cleaning answers ...")
        answer_df, detailed_answer_df = self.clean_answers_from_dict(
            segmentation_df=segmentation_df,
            question_answer_dict=self.get_default_question_answer_dict(
                segmentation_df, self.get_indicator_list(competence)
            ),
            indicator_value_dict=indicator_value_dict,
            year=year,
            textpages=None,
        )
        print("Done")
        print(f"Saving answers for {pdf_file} ...")
        self.save_answers(answer_df, detailed_answer_df, pdf_file, competence, year)
        print("Done")

    def run_cleaning_step(
        self,
        data_to_clean_benchmark_version: str,
        pdf_file: str,
        competence: str,
        year: str,
        remove_hallucinations=True,
    ):
        """
        Execute only the cleaning step on answers stored in the folder
        "/data/output/" + data_to_clean_benchmark_version + "/answers/"
        Cleaned answers are saved in self.answer_path
        """
        assert data_to_clean_benchmark_version != self.benchmark_version
        # Extract text (needed for removing hallucinations)
        if remove_hallucinations:
            extract_tables = self.extract_tables
            self.extract_tables = False
            textpages, _, _, _ = self.extract_text_and_tables_from_pdf(pdf_file)
            self.extract_tables = extract_tables
        else:
            textpages = None
        # Clean answers
        answer_df, detailed_answer_df = self.clean_answers_from_file(
            data_to_clean_benchmark_version,
            detailed_answer_file=pdf_file.split(".")[0] + "_detailed_answers.csv",
            year=year,
            textpages=textpages,
        )
        # Save answers (detailed answers have not been modified)
        self.save_answers(answer_df, detailed_answer_df, pdf_file, competence, year)
        answer_path = Path(
            *(self.answer_path / (pdf_file.split(".")[0] + "_answers.csv")).parts[-5:]
        )
        print(f"Answers are saved in {answer_path}")


####################################
def merge_question_answer_dicts(d1: dict, d2: dict) -> dict:
    assert d1.keys() == d2.keys()
    assert all(isinstance(value, list) for value in d1.values())
    assert all(isinstance(value, list) for value in d2.values())

    merged_dict = {}
    for key in d1.keys():
        merged_dict[key] = d1[key] + d2[key]
    return merged_dict
