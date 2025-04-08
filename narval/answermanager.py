import ast
import random
import re

import numpy as np
import pandas as pd
from unidecode import unidecode

from narval.prompts import NO_ANSWER_TAG
from narval.utils import FileSystem

# Define the (local or S3) file system
fs = FileSystem()


class AnswerManager:
    def __init__(self, indic_bound_file_path: str):
        # Initialize the dataframe self.indic_bound_df
        # giving the boundaries for each indicator
        columns = [
            "code_ip",  # indicator code
            "min_warning_ip",  # Lower boundary below which a warning is raised in SISPEA
            "max_warning_ip",  # Upper boundary above which a warning is raised in SISPEA
            "min_critic_ip",  # Lower boundary below which an error is raised in SISPEA
            "max_critic_ip",  # Upper boundary above which an error is raised in SISPEA
        ]
        self.indic_bound_df = fs.read_csv_to_df(indic_bound_file_path, usecols=columns)
        # Initalize the dataframe self.detailed_answer_df
        # storing answers lists for each (question, keyword_regex)
        self.detailed_answer_df = None
        # Initialize the dataframe self_answer_df
        # storing answers at various steps for each indicator
        self.answer_df = None
        # Define the tag corresponding to no answer given by the NLP model
        self.no_answer_tag = NO_ANSWER_TAG
        # Define the source strings
        self.table_source_str = "source: table extraction"
        self.llm_source_str = "source: language model"
        self.default_source_str = "source: undefined"

    def init_detailed_answer_df_from_segmentation_df(
        self, segmentation_df: pd.DataFrame
    ):
        # Initalize the dataframe self.detailed_answer_df
        # with the segmentation_df containing the questions
        # with their keyword regex and the relevant passages in the text
        self.detailed_answer_df = segmentation_df.copy()
        self.detailed_answer_df["answer_list_from_language_model"] = (
            self.detailed_answer_df["relevant_pages"].apply(
                lambda x: ["Not yet asked" for i in x]
            )
        )
        self.detailed_answer_df["answer_list_from_tables"] = None

    # Fill self.detailed_answer_df for a given question with answer_list
    def fill_detailed_answer_df_once(self, question: str, answer_list: list[str]):
        # Locate the index of the row where the 'question' matches
        mask = self.detailed_answer_df["question"] == question

        # Check there is a single matching row
        matching_rows = mask.sum()

        if matching_rows == 0:
            raise ValueError(f"Question '{question}' not found in the dataframe.")
        if matching_rows > 1:
            raise ValueError(
                f"Multiple rows found for question '{question}'. Ensure the question is unique."
            )
        # Get the index of the single row that matches
        idx = self.detailed_answer_df.index[mask][0]
        # Update the 'answer_list' for the matched row
        self.detailed_answer_df.at[idx, "answer_list_from_language_model"] = answer_list

    # Fill self.detailed_answer_df with question_answer_dict
    def fill_detailed_answer_df(self, question_answer_dict):
        for question, answer_list in zip(*question_answer_dict.values()):
            self.fill_detailed_answer_df_once(question, answer_list)

    # Fill the column "answer_list_from_tables" of self.detailed_answer_df for a given indicator
    def fill_detailed_answer_df_from_tables_once(
        self, indicator: str, answer_list: list[str]
    ):
        # Locate the index of the row for the indicator
        mask1 = self.detailed_answer_df["indicator"] == indicator
        mask2 = ~self.detailed_answer_df["table_relevant_pages"].isna()
        mask = mask1 & mask2

        # Check there is a single matching row
        matching_rows = mask.sum()

        if matching_rows == 0:
            raise ValueError(f"Indicator '{indicator}' not found in the dataframe.")
        if matching_rows > 1:
            raise ValueError(f"Multiple rows found for indicator '{indicator}'.")
        # Get the index of the single row that matches
        idx = self.detailed_answer_df.index[mask][0]
        # Update the 'answer_list' for the matched row
        self.detailed_answer_df.at[idx, "answer_list_from_tables"] = answer_list

    # Fill the column "answer_list_from_tables" of self.detailed_answer_df with indicator_value_dict
    def fill_detailed_answer_df_from_tables(self, indicator_value_dict):
        for indicator, answer_list in zip(*indicator_value_dict.values()):
            self.fill_detailed_answer_df_from_tables_once(indicator, answer_list)

    # Add a column "answer_list" in self.detailed_answer_df obtained by
    # concatenation of "answer_list_from_language_model" and "answer_list_from_tables"
    def add_answer_list_column_in_detailed_answer_df(self):
        df = self.detailed_answer_df
        df["answer_list"] = df["answer_list_from_tables"].apply(
            lambda l: [(x, self.table_source_str) for x in l] if l is not None else []
        ) + df["answer_list_from_language_model"].apply(
            lambda l: [(x, self.llm_source_str) for x in l] if l is not None else []
        )

    # Build the detailed_answer_df from the segmentation_df and the question_answer_dict
    def build_detailed_answer_df(
        self,
        segmentation_df: pd.DataFrame,
        question_answer_dict: dict,
        indicator_value_dict: dict,
    ):
        self.init_detailed_answer_df_from_segmentation_df(segmentation_df)
        self.fill_detailed_answer_df(question_answer_dict)
        self.fill_detailed_answer_df_from_tables(indicator_value_dict)
        self.add_answer_list_column_in_detailed_answer_df()

    # Get the detailed_answer_df from a file
    def get_detailed_answer_df_from_file(self, detailed_answer_file_path: str):
        columns = [
            "indicator",
            "question",
            "keyword_regex",
            "relevant_pages",
            "table_relevant_pages",
            "answer_list_from_language_model",
            "answer_list_from_tables",
            "answer_list",
        ]
        self.detailed_answer_df = fs.read_csv_to_df(
            detailed_answer_file_path, usecols=columns
        )
        # Convert the list strings back to lists
        liststring_cols = columns[3:]

        def convert_to_list(liststring):
            try:
                return ast.literal_eval(liststring)
            except ValueError:
                return np.nan

        for col in liststring_cols:
            self.detailed_answer_df.loc[:, col] = self.detailed_answer_df[col].apply(
                convert_to_list
            )

    # Build the dataframe self.answer_df with a column "indicator" and a column "concat_answer_list"
    # giving the list of answers for a given indicator
    # using self.detailed_answer_df
    # !!! This should be called after update of self.detailed_answer_df !!!
    def build_answer_df(self):
        # Group by 'indicator' and concatenate the 'answer_list'
        self.answer_df = (
            self.detailed_answer_df.groupby("indicator", as_index=False)["answer_list"]
            .apply(
                lambda x: [
                    item
                    for sublist in x
                    # if sublist != "Not yet asked"
                    for item in sublist
                ]
            )
            .rename({"answer_list": "concat_answer_list"}, axis=1)
        )

    # Clean the dataframe self.answer_df created with build_answer_df()
    # A new column "clean_answer_list" is added
    # by extracted numbers from the colum "concat_answer_list" (using regex)
    # except irrelevant numbers
    def clean_answers(self):
        # Define a function to clean each answer list with regex
        def clean_answer_with_regex(answer):
            # For the test answer "je crois  2.20. que 252.3 va D123 alors textee# P245.3 ok! 17. B415.3 €128."
            # this should return [252.3, 17.0, 128.0]
            # Removes spaces for millions eg "1 253 457" -> "1253457"
            answer = re.sub(
                r"(?<!\d)\b(\d{1,3}) (\d{3}) (\d{3})\b(?!\d)", r"\1\2\3", answer
            )
            # Remove spaces for eg "abc 12 345.7 xyz" -> "12345.7" but not "123 45"
            answer = re.sub(
                r"(?<!\d)\b(\d{1,}) (\d{3,})([.,]?\d{0,3})\b(?!\d)", r"\1\2\3", answer
            )
            # Remove spaces in fractions (eg "57 / 120" -> 57/120)
            answer = re.sub(r"\s?/\s?", r"/", answer)
            # Keep only the number of points for indicators in points (P202.2B, P255.3, ...)
            # Eg "70 points sur un total de 120" -> 70
            matches = re.findall(
                r"\D*(\d{1,3})(?: points)? sur\D* (?:100|120)\D*", answer
            )
            if len(matches) != 0:
                answer = " ".join(list(matches))
            # Remove all "." that are followed by a space and a capital letter
            answer = re.sub(r"\.(?=\s[A-Z])", "", answer)
            # Remove "." at the end of a sentence
            answer = re.sub(r"\.$", "", answer)
            # Split the string by spaces
            item_list = answer.split(" ")
            # Keep only items that contain at least one digit
            item_list = [item for item in item_list if re.search(r"\d", item)]
            # Keep only numerators from fractions over 100 or 120 (eg "57/120" -> 57 or "57.0/100" -> 57)
            # Relevant for indicators counted as points over 100 or 120
            item_list = [
                re.sub(r"^(\d+)([.,]0)?/(100|120)$", r"\1", item) for item in item_list
            ]
            # Same for indicators expressed as unitless numbers over 100 (eg km) or 1000 (eg abonnés)
            # eg 70.5u/100km -> 70.5
            item_list = [
                re.sub(r"^(\d+)([.,]\d*)?(\D{0,2})/(100|1000)\D*$", r"\1\2", item)
                for item in item_list
            ]
            # For other fractions, separate the numerator and denominator 'eg "22.4/75.2" -> ["22.4", "75.2"]
            item_list = [
                re.sub(r"(^\d+[.,]?\d+)/(\d+[.,]?\d+)$", r"\1##&\2", item)
                for item in item_list
            ]
            item_list = [item.split("##&") for item in item_list]
            # Flatten the list
            item_list = sum(item_list, [])
            # Remove punctuation characters except "," and "."
            item_list = [
                re.sub(r'[!"#$%&\'()*+:;<=>?@\[\\\]^_`{|}~]', "", item)
                for item in item_list
            ]
            # Remove items that have more than one period
            item_list = [item for item in item_list if item.count(".") <= 1]
            item_list = [item for item in item_list if item.count(",") <= 1]
            # Remove items that start with "D" or "P" (corresponding to indicator codes)
            item_list = [item for item in item_list if not item.startswith(("D", "P"))]
            # Remove special items m2 and m3
            item_list = [
                re.sub(r"(\w*)m[23](?:/\w+)?\b", r"\1", item) for item in item_list
            ]
            # Extract digits and decimal point
            item_list = [re.sub(r"[^\d.,]", "", item) for item in item_list]
            # Replace "," by "."
            item_list = [re.sub(r"(?<=\d),(?=\d|$)", ".", item) for item in item_list]
            # Remove empty strings
            item_list = [
                item for item in item_list if item.strip() not in ["", ",", "."]
            ]
            # Convert to float
            item_list = [float(item) for item in item_list]

            return item_list

        def clean_list(answer_list):
            cleaned_list = []
            for answer, source in answer_list:
                # If "je ne trouve pas" is in answer, keep it
                if re.search(
                    self.no_answer_tag, answer, re.IGNORECASE
                ) and self.no_answer_tag not in [x[0] for x in cleaned_list]:
                    cleaned_list.append((self.no_answer_tag, source))
                elif answer == "Not yet asked" and answer not in [
                    x[0] for x in cleaned_list
                ]:
                    cleaned_list.append((answer, source))
                else:
                    try:
                        numbers = clean_answer_with_regex(answer)
                        cleaned_list.extend([(nb, source) for nb in numbers])
                    except ValueError:
                        print(
                            f"The answer '{answer}' could not be cleaned and has been ignored"
                        )
            return cleaned_list

        # Apply the cleaning function to each 'answer_list'
        clean_answer_list = self.answer_df["concat_answer_list"].apply(clean_list)
        self.answer_df.insert(1, "clean_answer_list", clean_answer_list)

    # Re-Clean the column 'clean_answer_list' of self.answer_df
    # by removing floats in forbidden_number_list
    # and additional floats defined below per indicator (not well written ...)
    def remove_forbidden_numbers(self, forbidden_number_list=None):
        if forbidden_number_list is None:
            forbidden_number_list = []

        def remove_irrelevant_numbers(row, forbidden_number_list):
            # Define forbidden numbers for each indicator
            unwanted_nb_dict = {
                # D204.0 = "price per m3 for an annual consumption of 120 m3"
                # but "120" is not a relevant answer
                "D204.0": [120],
                # Add more
                "P252.2": [100],
                "P251.1": [1000],
                "P258.1": [1000],
            }
            forbidden_nb_per_indic_dict = {
                key: [] for key in self.indic_bound_df["code_ip"]
            }
            forbidden_nb_per_indic_dict.update(unwanted_nb_dict)
            # Update the list of forbidden numbers
            new_forbidden_number_list = (
                forbidden_number_list + forbidden_nb_per_indic_dict[row["indicator"]]
            )
            # Remove forbidden numbers from clean_answer_list
            clean_answer_list = row["clean_answer_list"]
            row["clean_answer_list"] = [
                x for x in clean_answer_list if x[0] not in new_forbidden_number_list
            ]

            return row

        # Update the column "clean_answer_list" in the dataframe self.answer_df
        self.answer_df = self.answer_df.apply(
            # remove_irrelevant_numbers, args=forbidden_number_list, axis=1
            lambda row: remove_irrelevant_numbers(row, forbidden_number_list),
            axis=1,
        )

    # Re-Clean the column 'clean_answer_list' of self.answer_df
    # by removing hallucinations ie numbers from clean_answer_list
    # that do not appear in the pages from textpages
    # that are relevant to the corresponding indicator
    # Note the input parameter textpages must correspond to the full PDF text pages
    def remove_hallucinations(self, textpages: list[str]):
        def is_hallucination(answer, textpage):
            if answer == self.no_answer_tag or answer == "Not yet asked":
                return False

            assert isinstance(answer, float)

            # 0 and 100 answers corresponding to "aucune autorisation" and "conforme"
            # should not be considered as hallucinations
            if answer == 0 or answer == 100:
                return False
            # Removes spaces for millions eg "1 253 457" -> "1253457"
            textpage = re.sub(
                r"(?<!\d)\s*(\d{1,3}) (\d{3}) (\d{3})\b(?!\d)", r"\1\2\3", textpage
            )
            # Remove spaces for eg "12 345" -> "12345" but not "123 45"
            textpage = re.sub(
                r"(?<!\d)\s*(\d{1,2}) (\d{3,})\b(?!\d)", r"\1\2", textpage
            )
            # Remove spaces for eg "abc 1 024 2 148 xyz" -> "1024 2148"
            textpage = re.sub(
                r"(?<!\d)\s*(\d{1,2}) (\d{3,}) +(\d{1,2}) (\d{3,})\b(?!\d)",
                r"\1\2 \3\4",
                textpage,
            )
            # Format textpage
            textpage = unidecode(textpage)
            # Remove special items m2 or m3
            textpage = re.sub(r"(\w*)m[23](?:/\w+)?\b", r"\1", textpage)
            # Remove dates
            textpage = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", "", textpage)
            # Generate possible patterns for the answer
            patterns = []
            patterns.append(str(answer))
            if "." in str(answer):
                patterns.append(str(answer).replace(".", ","))
                # Remove trailing zeros eg 12.0 -> 12
                patterns.append(str(answer).rstrip("0").rstrip("."))

            patterns = list(set(patterns))
            # Build the regex
            patterns = [
                (
                    r"(?<!\d|,|\.)"
                    + re.escape(p)
                    + r"(?=([.,]0{1,2})\b|[^.,\d]|[.]?$|[.]?\s)"
                    if "." not in p and "," not in p
                    else r"(?<!\d|,|\.)"
                    + re.escape(p)
                    + r"(?=[^.,123456789]|[.]?$|[.]?\s)"
                )
                for p in patterns
            ]
            regex = "|".join(patterns)

            return not bool(re.search(regex, textpage))

        def remove_hallu(row):
            clean_answer_list = row["clean_answer_list"]
            indicator = row["indicator"]
            # Get the text pages extracted for this indicator
            # In principle, each raw answer was obtained from a single page
            # and one should give only one page per answer to the function is_hallucination
            # But retrieving this page number is tricky for the present cleaning pipeline
            # To be improved ...
            relevant_pages = (
                self.detailed_answer_df.query("indicator==@indicator")
                .explode("relevant_pages")
                .relevant_pages.dropna()
                .unique()
            )
            concat_textpage = "".join([textpages[i] for i in relevant_pages])
            # Remove hallucinations from clean_answer_list
            # (not needed when answers are extracted from tables)
            row["clean_answer_list"] = [
                x
                for x in clean_answer_list
                if x[1] == self.table_source_str
                or not is_hallucination(x[0], concat_textpage)
            ]

            return row

        # Update the column "clean_answer_list" in the dataframe self.answer_df
        self.answer_df = self.answer_df.apply(remove_hallu, axis=1)

    # Filter answer_list within [min, max]
    # Keep also str values
    def filter_answer_list_oob(self, answer_list, mini=0, maxi=10**6):
        float_list = [x for x in answer_list if isinstance(x[0], float)]
        str_list = [x for x in answer_list if isinstance(x[0], str)]
        new_answer_list = [x for x in float_list if mini <= x[0] <= maxi] + str_list

        return new_answer_list

    # Add a new column "filtered_answer_list" to self.answer_df
    # by removing answers from the colum "clean_answer_list"
    # that are out of indicator boundaries (oob)
    def exclude_critic_oob_answers(self):
        # Check that clean_answers() has already been called
        assert "clean_answer_list" in self.answer_df.columns

        # Filter clean_answer_list based on min and max values of each indicator
        # Keep also str values
        def filter_answer_list(row):
            answer_list = row["clean_answer_list"].copy()
            min_ip = row["min_critic_ip"]
            max_ip = row["max_critic_ip"]

            return self.filter_answer_list_oob(answer_list, mini=min_ip, maxi=max_ip)

        new_answer_list = self.answer_df.merge(
            self.indic_bound_df, left_on="indicator", right_on="code_ip", how="left"
        ).apply(filter_answer_list, axis=1)
        self.answer_df.insert(1, "filtered_answer_list", new_answer_list)

    # Add a new column "final_answer" to self.answer_df
    # by selecting for each row one answer from the colum "filtered_answer_list"
    def select_one_answer_per_indic(self, seed=1):
        # Check that exclude_critic_oob_answers() has already been run
        assert "filtered_answer_list" in self.answer_df.columns
        # Fix the seed
        random.seed(seed)

        def select_most_frequent_values(float_list):
            float_array = np.array(float_list)
            # Get unique values and their counts
            unique_values, counts = np.unique(float_array, return_counts=True)
            # Find the maximum frequency
            max_freq = np.max(counts)
            # Extract the values that have the maximum frequency
            most_frequent_values_array = unique_values[counts == max_freq]

            return most_frequent_values_array

        def choose_answer_in_list(row):
            answer_list = row["filtered_answer_list"].copy()
            # If empty answer_list
            if len(answer_list) == 0:
                answer = self.no_answer_tag
                source = self.default_source_str
            # If the NLP model has not been asked to give an answer
            elif len(answer_list) == 1 and answer_list[0][0] == "Not yet asked":
                answer = "Not yet asked"
                source = self.default_source_str
            # If the NLP has been asked to give an answer but returns no_answer_tag
            elif len(answer_list) == 1 and answer_list[0][0] == self.no_answer_tag:
                answer = self.no_answer_tag
                source = self.llm_source_str
            # otherwise
            else:
                # Keep only floats
                answer_list = [x for x in answer_list if isinstance(x[0], float)]
                # Select value(s) from answer_list with the highest frequency
                top_answers = select_most_frequent_values([x[0] for x in answer_list])
                answer_list = [x for x in answer_list if x[0] in top_answers]
                # Remove answers outside sispea warning boundaries
                # except if all of them are outside boundaries
                # In that case, keep them all
                if len(answer_list) > 1:
                    test_answer_list = self.filter_answer_list_oob(
                        answer_list,
                        mini=row["min_warning_ip"],
                        maxi=row["max_warning_ip"],
                    )
                    if len(test_answer_list) > 0:
                        answer_list = test_answer_list
                # Randomly choose one of the remaining values
                if len(answer_list) > 1:
                    answer, source = random.choice(answer_list)
                else:
                    answer, source = answer_list[0]

            return (answer, source)

        # Apply the selector
        answers = self.answer_df.merge(
            self.indic_bound_df, left_on="indicator", right_on="code_ip", how="left"
        ).apply(choose_answer_in_list, axis=1)
        self.answer_df.insert(1, "final_answer_source", [x[1] for x in answers])
        self.answer_df.insert(1, "final_answer", [x[0] for x in answers])
        # Clean the source column
        self.answer_df["final_answer_source"] = (
            self.answer_df["final_answer_source"]
            .str.replace(r"source:", "", regex=True)
            .str.strip()
        )

    def apply_full_cleaning_pipeline(self, textpages=None, forbidden_number_list=None):
        self.build_answer_df()
        self.clean_answers()
        if textpages is not None:
            self.remove_hallucinations(textpages)
        self.remove_forbidden_numbers(forbidden_number_list=forbidden_number_list)
        self.exclude_critic_oob_answers()
        self.select_one_answer_per_indic()
