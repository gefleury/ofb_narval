import re

import pandas as pd

from narval.utils import FileSystem

# Define the (local or S3) file system
fs = FileSystem()


class PageFinder:
    def __init__(self, question_keyword_file_path: str, competence: str):
        assert competence in [
            "eau potable",
            "assainissement collectif",
            "assainissement non collectif",
        ]

        self.keyword_df = fs.read_csv_to_df(
            question_keyword_file_path, index_col=0
        ).query("competence==@competence")

        self.segmentation_df = None

    def init_segmentation_df(self):
        self.segmentation_df = (
            self.keyword_df
            # Modify the keyword into a regex to extract only words and not subwords , with or without an "s"
            .assign(mot=lambda df: r"\b" + df["mot"] + r"s?\b")
            .groupby(["indic", "question"], as_index=False)["mot"]
            # Join keywords if there a several keywords for one question
            .agg(r"|".join)
            .rename({"mot": "keyword_regex", "indic": "indicator"}, axis=1)
        )

    def add_text_relevant_pages_column_to_segmentation_df(
        self, text_pages: list, excluded_page_indices=None
    ):
        """
        Add a column "relevant pages" to self.segmentation_df
        giving for each question the relevant pages numbers in the text_pages list

        Parameters
        ----------
        text_pages: list of str
            Text of page i is contained in text_pages[i] (pages are counted from 0)
        excluded_page_indices: list of int (indices of pages that must not be extracted)
        """
        # Apply the regex pattern for each question across all pages
        text_series = pd.Series(text_pages)
        self.segmentation_df["relevant_pages"] = self.segmentation_df[
            "keyword_regex"
        ].apply(
            lambda pattern: text_series[
                text_series.str.contains(pattern, case=False, regex=True, na=False)
            ].index.tolist()
        )

        # Remove forbidden pages
        if excluded_page_indices is None:
            excluded_page_indices = []
        self.segmentation_df["relevant_pages"] = self.segmentation_df[
            "relevant_pages"
        ].apply(lambda l: [idx for idx in l if idx not in excluded_page_indices])

    def add_table_relevant_pages_column_to_segmentation_df(self, df_list_list: list):
        """
        Add a column "table_relevant_pages" to self.segmentation_df
        giving for each indicator the location of relevant tables in df_list_list
        Only tables containing the indicator code (outside the header) are considered are relevant

        Parameters
        ----------
        df_list_list: list of list of pandas dataframes
            Table j on page i is contained in df_list_list[i][j]
        """

        def is_regex_in_df(df, str_pattern):
            regex_pattern = re.compile(str_pattern)
            return bool(re.search(regex_pattern, df.to_string(header=False)))

        def find_table_indices(str_pattern):
            bool_list_list = [
                [is_regex_in_df(df, str_pattern) for df in df_list]
                for df_list in df_list_list
            ]
            # Collect [sub_list index, element_index] for each True value
            idx_list_list = [
                [i, j]
                for i, sublist in enumerate(bool_list_list)
                for j, value in enumerate(sublist)
                if value
            ]
            return idx_list_list

        df = self.segmentation_df[["indicator", "keyword_regex"]]
        df = df[df["keyword_regex"] == r"\b" + df["indicator"] + r"s?\b"]
        df["table_relevant_pages"] = df["keyword_regex"].apply(find_table_indices)
        self.segmentation_df = self.segmentation_df.merge(
            df, on=["indicator", "keyword_regex"], how="left"
        )

    def build_segmentation_df(
        self, text_pages: list, df_list_list: list, excluded_page_indices=None
    ):
        self.init_segmentation_df()
        self.add_text_relevant_pages_column_to_segmentation_df(
            text_pages, excluded_page_indices
        )
        self.add_table_relevant_pages_column_to_segmentation_df(df_list_list)

    def get_segmentation_df(
        self, text_pages: list, df_list_list: list, excluded_page_indices=None
    ):
        self.build_segmentation_df(text_pages, df_list_list, excluded_page_indices)
        return self.segmentation_df
