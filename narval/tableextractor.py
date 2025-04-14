import warnings

import pandas as pd

from narval.utils import FileSystem

# Define the (local or S3) file system
fs = FileSystem()


class TableExtractor:
    def __init__(self, df: pd.DataFrame, question_file_path: str, year: str):
        # List of searched indicators
        question_df = fs.read_csv_to_df(question_file_path, index_col=0)
        self.indicator_list = question_df["indic"].unique().tolist()
        # Year
        self.year = year
        # Raw dataframe
        self.df = df.copy()
        self.clean_df_header()
        # Cleaned sub dataframe with indicator codes and values
        self.sub_df = None
        self.extract_and_clean_sub_df()

    def clean_df_header(self):
        """
        Clean the header of self.df
        Should be improved
        """
        if any(col.startswith("Col_") for col in self.df.columns):
            # Concatenate the header with the first row
            self.df.columns = self.df.columns + "_" + self.df.iloc[0].fillna("")
            # And remove the first row
            self.df.drop(index=self.df.index[0], axis=0, inplace=True)

    def find_indicator_code_column(self):
        # Extract the column with indicator codes
        code_regex = r"|".join(self.indicator_list)
        code_col_df = self.df.apply(
            lambda col: col.astype("str")
            .str.contains(code_regex, na=False, regex=True)
            .any(),
            axis=0,
        )
        code_col = code_col_df[code_col_df].index.tolist()
        if code_col:  # no empty list
            # Check all indicator codes are in the same column
            # (typical structure of summary tables)
            assert len(code_col) == 1
            code_col = code_col[0]
        else:  # there is no column for indicator codes
            code_col = None

        return code_col

    def find_indicator_value_column_from_header(self):
        # Search a column with year in its name but not year-1
        # by searching in the header
        value_col = [
            col
            for col in self.df.columns
            if str(self.year) in col and str(int(self.year) - 1) not in col
        ]

        if value_col:  # no empty list
            assert len(value_col) == 1
            value_col = value_col[0]
        else:  # there is no column for indicator values for this year
            value_col = None

        return value_col

    def find_indicator_value_column_from_row(self, row_index: int):
        """
        Returns the indicator value column containing year in its name but not year-1
        by searching in the row at index "row_index"
        """
        # If there is null values in the row,
        # it cannot be safely used to identify the indicator value column
        if self.df.iloc[row_index].isna().sum() > 0:
            return None

        value_col = [
            col
            for cell, col in zip(self.df.iloc[row_index], self.df.columns)
            if cell and str(self.year) in cell and str(int(self.year) - 1) not in cell
        ]

        if value_col:  # no empty list
            assert len(value_col) == 1
            value_col = value_col[0]
        else:  # there is no column for indicator values for this year
            value_col = None

        return value_col

    def find_indicator_value_column(self):
        # Search in the header
        value_col = self.find_indicator_value_column_from_header()
        if value_col:
            return value_col
        # If not, search in the first row
        value_col = self.find_indicator_value_column_from_row(row_index=0)

        return value_col

    def extract_sub_df(self):
        if not self.df.empty:
            # Extract the two relevant columns with indicator codes and values
            code_col = self.find_indicator_code_column()
            value_col = self.find_indicator_value_column()
            # Extract the corresponding sub_df
            if code_col and value_col and code_col != value_col:
                sub_df = self.df[[code_col, value_col]]
                sub_df = sub_df.dropna(axis=0, how="all")
                sub_df.columns = ["Indicator_code", "Indicator_value"]
                self.sub_df = sub_df

    def clean_sub_df(self):
        if self.is_summary_table():
            self.sub_df = trim_all_columns(self.sub_df)

            # Solve the problem of added rows when extracting tables with PDFPlumber
            # by merging consecutive rows corresponding to the same indicator eg for
            # D204  None
            # None  2.45
            self.sub_df.loc[:, "Indicator_code"] = self.sub_df["Indicator_code"].ffill()
            self.sub_df.loc[:, "Indicator_value"] = self.sub_df[
                "Indicator_value"
            ].fillna("")

            # Merge rows with the same indicator code
            # only if there is no more than 1 row containing digits in "Indicator value"
            # ... Count rows with digits per 'Indicator_code'
            digit_counts = self.sub_df.groupby("Indicator_code")[
                "Indicator_value"
            ].apply(lambda col: sum(contains_digits(s) for s in col))
            # ... Merge only if digit count is ≤ 1
            valid_codes = digit_counts[digit_counts <= 1].index
            merged_df = (
                self.sub_df.query("Indicator_code in @valid_codes")
                .groupby("Indicator_code", as_index=False, sort=False)
                .agg({"Indicator_value": lambda col: " ".join(col.dropna())})
            )
            # ... Combine merged and unmerged data
            self.sub_df = pd.concat(
                [
                    merged_df,
                    self.sub_df.query("Indicator_code not in @valid_codes"),
                ],
                ignore_index=True,
            )

    def extract_and_clean_sub_df(self):
        """
        Extract the clean self.sub_df from self.df
        made of the indicator code and value columns if both exist
        self.sub_df is None if self.df is not a summary table
        """
        self.extract_sub_df()
        self.clean_sub_df()

    def is_summary_table(self):
        """
        Returns True if self.df is a summary table else False.
        A summary table is a table having a column with indicator codes
        and a column with indicator values for the relevant year,
        independently of the number of indicators in the table
        """
        return self.sub_df is not None

    def extract_indicator_value(self, indicator_code: str):
        """
        Extract the value of indicator_code in self.sub_df
        """
        if not self.is_summary_table():
            return "Not found"
        # Check if indicator_code is in the dataframe
        if all(indicator_code not in x for x in self.sub_df["Indicator_code"].tolist()):
            return "Not found"
        # Check if indicator_code appears more than once
        if sum(indicator_code in x for x in self.sub_df["Indicator_code"].tolist()) > 1:
            warnings.warn(
                f"The indicator {indicator_code} appears more than once in the table"
            )
            return "Not found"
        # Query the row containing indicator_code
        matched_row = self.sub_df.query("Indicator_code.str.contains(@indicator_code)")
        # Check if other indicators from indicator_list are also present in the same row
        # (may happen if several rows are merged into one by PDFPlumber)
        if any(
            ind in matched_row.iloc[0]["Indicator_code"]
            for ind in self.indicator_list
            if ind != indicator_code
        ):
            return "Not found"

        # Extract the indicator value
        return matched_row.iloc[0]["Indicator_value"]


def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all columns in dataframe
    """
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)


def contains_digits(s: str):
    return any(char.isdigit() for char in str(s))
