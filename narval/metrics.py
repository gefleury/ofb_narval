import numpy as np
import pandas as pd

from narval.prompts import NO_ANSWER_TAG
from narval.utils import FileSystem, get_data_dir


class MetricsCalculator:
    def __init__(self):
        self.fs = FileSystem()
        self.dir = get_data_dir()
        self.path_to_ref_values = (
            self.dir + "/data/input/sispea_vs_pdf_indic_values/v2/"
        )
        self.path_per_pdf = self.dir + "/data/output/" + "all_metrics_per_pdf.csv"
        self.path_per_indic = self.dir + "/data/output/" + "all_metrics_per_indic.csv"

        try:
            self.df_per_pdf = self.fs.read_csv_to_df(self.path_per_pdf)
        except FileNotFoundError:
            columns = [
                "benchmark_version",
                "year",
                "competence",
                "pdf_name",
                "no_value_indic_nb_in_sispea",
                "accuracy_vs_sispea",
                "precision_vs_sispea",
                "recall_vs_sispea",
                "tp_nb_vs_sispea",
                "tn_nb_vs_sispea",
                "fp1_nb_vs_sispea",
                "fp2_nb_vs_sispea",
                "fn_nb_vs_sispea",
                "no_value_indic_nb_in_pdf",
                "accuracy_vs_pdf",
                "precision_vs_pdf",
                "recall_vs_pdf",
                "tp_nb_vs_pdf",
                "tn_nb_vs_pdf",
                "fp1_nb_vs_pdf",
                "fp2_nb_vs_pdf",
                "fn_nb_vs_pdf",
            ]
            self.df_per_pdf = pd.DataFrame(columns=columns)

        try:
            self.df_per_indic = self.fs.read_csv_to_df(self.path_per_indic)
        except FileNotFoundError:
            columns = [
                "benchmark_version",
                "pdf_list_file",
                "indicator",
                "no_value_indic_nb_in_sispea",
                "accuracy_vs_sispea",
                "precision_vs_sispea",
                "recall_vs_sispea",
                "tp_nb_vs_sispea",
                "tn_nb_vs_sispea",
                "fp1_nb_vs_sispea",
                "fp2_nb_vs_sispea",
                "fn_nb_vs_sispea",
                "no_value_indic_nb_in_pdf",
                "accuracy_vs_pdf",
                "precision_vs_pdf",
                "recall_vs_pdf",
                "tp_nb_vs_pdf",
                "tn_nb_vs_pdf",
                "fp1_nb_vs_pdf",
                "fp2_nb_vs_pdf",
                "fn_nb_vs_pdf",
            ]
            self.df_per_indic = pd.DataFrame(columns=columns)

    def write_answers_vs_true_file(self, answer_file: str, benchmark_version: str):
        """
        Write a new csv file containing answers together with
        true indicator values from sispea and from pdf labeling

        Parameters
        ----------
        answer_file : str
            The name of the file containing model answers
        benchmark_version: str
            The benchmark version used in the folder tree
        """
        answer_path = self.dir + "/data/output/" + benchmark_version + "/answers/"
        output_file = answer_file.split(".")[0] + "_vs_true" + ".csv"
        # Get the answer dataframe
        df = self.fs.read_csv_to_df(answer_path + answer_file)
        # Get the path of the file with true indicator values
        true_val_file = answer_file.split(".")[0]
        true_val_file = "_".join(true_val_file.split("_")[1:])
        true_val_file = (
            true_val_file.replace("_answers", "_sispea_vs_pdf_indic_values") + ".csv"
        )
        true_val_path = self.path_to_ref_values + true_val_file
        # Add a colum in df with true indicator values (from sispea)
        df2 = self.fs.read_csv_to_df(
            true_val_path, usecols=["indicator", "sispea_value"]
        )
        true_indic_list = [
            df2.loc[df2["indicator"] == indic, "sispea_value"].values[0]
            for indic in df.indicator
        ]
        df.insert(5, "true_sispea_value", true_indic_list)
        # Add a colum in df with true indicator values (from pdf labeling)
        try:
            df2 = self.fs.read_csv_to_df(
                true_val_path, usecols=["indicator", "pdf_value"]
            )
            true_indic_list = [
                df2.loc[df2["indicator"] == indic, "pdf_value"].values[0]
                for indic in df.indicator
            ]
            df.insert(6, "true_pdf_value", true_indic_list)
        except ValueError:  # if the pdf has not yet been labeled
            true_indic_list = ["not yet labeled" for indic in df.indicator]
            df.insert(6, "true_pdf_value", true_indic_list)
        # Save df
        self.fs.write_df_to_csv(df, answer_path + output_file, index=False)

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute metrics (using two arrays of true and predicted values)
        - the number of true positives (#TP).
            A pred value is counted as a true positive
            when it is a number equal to the true value
        - the number of true negatives (#TN).
            A pred value is counted as a true negative (#TN)
            when it is np.nan and the true value is also np.nan
        - the number of false positives of kind 1 (#FP1)
            A pred value is counted as a false positive of kind 1 (#FP1)
            when it is a number while the true value is np.nan
        - the number of false positives of kind 2 (#FP2)
            A pred value is counted as a false positive of kind 2 (#FP2)
            when it is a number while the true value is another number
        - the number of false negatives (#FN)
            A pred value is counted as a false negative (#FN)
            when it is np.nan while the true value is a number
        - accuracy
        - precision
        - recall
        - number of Null (np.nan) values in y_true

        Parameters
        ----------
        y_true : array-like
            The array of true values. This can contain floats or `np.nan` values.
        y_pred : array-like
            The array of predicted values, with the same shape as `y_true`.
            This can also contain floats or `np.nan` values.

        Returns
        -------
        A tuple with all metrics
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
        assert len(y_true) == len(y_pred)
        assert len(y_true) > 0

        tp_number = np.sum((y_true == y_pred) & (~np.isnan(y_true)))
        tn_number = np.sum((np.isnan(y_pred)) & (np.isnan(y_true)))
        fp1_number = np.sum((np.isnan(y_true)) & (~np.isnan(y_pred)))
        fp2_number = np.sum(
            (y_true != y_pred) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        )
        fn_number = np.sum((~np.isnan(y_true)) & (np.isnan(y_pred)))
        accuracy = (tp_number + tn_number) / len(y_pred)
        precision = np.divide(
            tp_number, np.sum(~np.isnan(y_pred)), where=np.sum(~np.isnan(y_pred)) != 0
        )
        recall = np.divide(
            tp_number, np.sum(~np.isnan(y_true)), where=np.sum(~np.isnan(y_true)) != 0
        )
        true_null_number = np.sum(np.isnan(y_true))

        # Basic check
        assert tp_number + tn_number + fp1_number + fp2_number + fn_number == len(
            y_true
        )
        assert np.sum(~np.isnan(y_pred)) == (tp_number + fp1_number + fp2_number)
        assert np.sum(~np.isnan(y_true)) == (tp_number + fn_number + fp2_number)
        assert true_null_number == tn_number + fp1_number

        return (
            true_null_number,
            accuracy,
            precision,
            recall,
            tp_number,
            tn_number,
            fp1_number,
            fp2_number,
            fn_number,
        )

    def get_df_from_answer_file(
        self, answer_vs_true_file: str, benchmark_version: str
    ) -> pd.DataFrame:
        # Get the dataframe with answers and true indicator values
        answer_path = f"{self.dir}/data/output/{benchmark_version}/answers/"
        df = self.fs.read_csv_to_df(answer_path + answer_vs_true_file)
        # Drop useless columns
        df = df.drop(
            [
                "filtered_answer_list",
                "clean_answer_list",
                "concat_answer_list",
            ],
            axis=1,
        )

        return df

    def compute_metrics_df_per_pdf_helper(
        self, answer_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes the metrics dataframe (one row) from answer_df

        Parameters
        ----------
        answer_df : pd.DataFrame
            The answer dataframe for one PDF, one benchmark version

        Returns
        -------
        row_df : a dataframe with a single row
        """
        # Check the input answer_df corresponds to
        # a single PDF and benchmark version
        cols = ["benchmark_version", "pdf_name"]
        assert len(answer_df[cols].drop_duplicates() == 1)

        # Prepare the metrics dataframe
        df = answer_df.copy()
        columns = [
            "benchmark_version",
            "year",
            "competence",
            "pdf_name",
        ]
        row_df = df[columns].drop_duplicates()
        assert len(row_df) == 1

        # Fill the metrics dataframe
        y_pred = pd.to_numeric(
            df["final_answer"].replace(NO_ANSWER_TAG, np.nan),
            errors="coerce",
        )
        for ref in ["sispea", "pdf"]:
            y_true = df[f"true_{ref}_value"]
            if y_true.isin(["Not yet searched", "not yet labeled"]).any():
                # If no sispea/pdf labeling, the metrics cannot be calculated
                metrics = [np.nan] * 9
            else:
                metrics = self.compute_metrics(y_true.values, y_pred.values)

            row_df[f"no_value_indic_nb_in_{ref}"] = metrics[0]
            row_df[f"accuracy_vs_{ref}"] = metrics[1]
            row_df[f"precision_vs_{ref}"] = metrics[2]
            row_df[f"recall_vs_{ref}"] = metrics[3]
            row_df[f"tp_nb_vs_{ref}"] = metrics[4]
            row_df[f"tn_nb_vs_{ref}"] = metrics[5]
            row_df[f"fp1_nb_vs_{ref}"] = metrics[6]
            row_df[f"fp2_nb_vs_{ref}"] = metrics[7]
            row_df[f"fn_nb_vs_{ref}"] = metrics[8]

        return row_df

    def compute_metrics_df_per_pdf(
        self, answer_vs_true_file: str, benchmark_version: str
    ) -> pd.DataFrame:
        """
        Computes the metrics dataframe for the input file

        Parameters
        ----------
        answer_vs_true_file : str
            The name of the file containing answers and true indicator values for given pdf
        benchmark_version: str
            The benchmark version used in the folder tree

        Returns
        -------
        res_df : a dataframe with 1 row
        """
        answer_df = self.get_df_from_answer_file(answer_vs_true_file, benchmark_version)
        res_df = self.compute_metrics_df_per_pdf_helper(answer_df)

        return res_df

    def fill_metrics_df_per_pdf(self, answer_vs_true_file: str, benchmark_version: str):
        """
        Add 1 row in the dataframe self.df_per_pdf with metrics calculated for pdf_name

        Parameters
        ----------
        answer_vs_true_file : str
            The name of the file containing answers and true indicator values for given pdf
        benchmark_version: str
            The benchmark version used in the folder tree
        """
        df = self.compute_metrics_df_per_pdf(answer_vs_true_file, benchmark_version)
        self.df_per_pdf = pd.concat(
            [self.df_per_pdf if not self.df_per_pdf.empty else None, df],
            axis=0,
            ignore_index=True,
        )
        # Remove duplicates
        columns = [
            "benchmark_version",
            "year",
            "competence",
            "pdf_name",
        ]
        self.df_per_pdf = self.df_per_pdf.drop_duplicates(subset=columns)

    def compute_metrics_df_per_indic_helper(
        self, answer_df: pd.DataFrame
    ) -> pd.DataFrame:
        df = answer_df.copy()

        # Check the input answer_df corresponds to
        # a single benchmark version
        cols = ["benchmark_version"]
        assert len(answer_df[cols].drop_duplicates() == 1)

        # Replace NO_ANSWER_TAG by nan in the column "final_answer"
        df["final_answer"] = pd.to_numeric(
            df["final_answer"].replace(NO_ANSWER_TAG, np.nan),
            errors="coerce",
        )
        # Compute the metrics per indicator
        df = df.groupby(["benchmark_version", "indicator"], as_index=False).agg(list)
        df = df.rename({"pdf_name": "pdf_name_list"}, axis=1)
        # ... Add 9 columns for the metrics vs sispea or vs pdf
        for ref in ["pdf", "sispea"]:
            new_columns = [
                f"no_value_indic_nb_in_{ref}",
                f"accuracy_vs_{ref}",
                f"precision_vs_{ref}",
                f"recall_vs_{ref}",
                f"tp_nb_vs_{ref}",
                f"tn_nb_vs_{ref}",
                f"fp1_nb_vs_{ref}",
                f"fp2_nb_vs_{ref}",
                f"fn_nb_vs_{ref}",
            ]
            df[new_columns] = df[[f"true_{ref}_value", "final_answer"]].apply(
                lambda row, ref_val=f"true_{ref}_value": (
                    # if no reference value, the metrics cannot be calculated
                    [np.nan for i in range(9)]
                    if any(
                        x in row[ref_val]
                        for x in ["Not yet searched", "not yet labeled"]
                    )
                    else self.compute_metrics(
                        np.array(row[ref_val]),
                        np.array(row["final_answer"]),
                    )
                ),
                result_type="expand",
                axis=1,
            )
        # Keep only relevant columns
        df = df.drop(
            ["true_sispea_value", "true_pdf_value", "final_answer"],
            axis=1,
        )

        return df

    def compute_metrics_df_per_indic(
        self, pdf_list: list[str], benchmark_version: str
    ) -> pd.DataFrame:
        """
        Compute the metrics per indicator for all pdfs in pdf_list

        Parameters
        ----------
        pdf_list : list[str]
            List of pdf names
        benchmark_version: str
            The benchmark version used in the folder tree

        Returns
        -------
        A dataframe with metrics per indicator calculated on the bunch of pdfs
        """

        # Concatenate all dataframes with answers and true indicator values
        # for pdfs in pdf_list
        answer_df = pd.DataFrame()

        for pdf_name in pdf_list:
            answer_vs_true_file = pdf_name.split(".")[0] + "_answers_vs_true.csv"
            try:
                df_one_pdf = self.get_df_from_answer_file(
                    answer_vs_true_file, benchmark_version
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"The answers have not yet been calculated for the pdf {pdf_name}.pdf and {benchmark_version}"
                ) from e
            else:
                answer_df = pd.concat([answer_df, df_one_pdf], axis=0)

        answer_df = answer_df.drop(["year", "competence"], axis=1)

        # Compute metrics
        res_df = self.compute_metrics_df_per_indic_helper(answer_df)

        return res_df

    def fill_metrics_df_per_indic(self, pdf_list_file: str, benchmark_version: str):
        """
        Update the dataframe self.df_per_indic with metrics per indicator
        calculated on the bunch of PDFs listed in pdf_list_file

        Parameters
        ----------
        pdf_list_file : str
            The name of the file containing pdfs names on which metrics shall be calculated
            Warning : this is a csv file with ";" as separator
        benchmark_version: str
            The benchmark version used in the folder tree
        """
        # Get the list of pdf names
        eval_path = "/data/input/" + pdf_list_file
        eval_df = self.fs.read_csv_to_df(
            self.dir + eval_path, sep=";", usecols=["pdf_name"]
        )
        pdf_list = eval_df["pdf_name"].values.tolist()
        # Compute the metrics per indicator
        df = self.compute_metrics_df_per_indic(pdf_list, benchmark_version)
        df["pdf_list_file"] = pdf_list_file
        # Fill the metrics dataframe self.df_per_indic
        self.df_per_indic = pd.concat(
            [
                self.df_per_indic if not self.df_per_indic.empty else None,
                df[self.df_per_indic.columns],
            ],
            axis=0,
            ignore_index=True,
        )
        # Remove duplicates
        columns = ["benchmark_version", "pdf_list_file", "indicator"]
        self.df_per_indic = self.df_per_indic.drop_duplicates(subset=columns)

    def compute_average_metrics(self) -> pd.DataFrame:
        """
        Quick and dirty way to compute the average metrics from self.df_per_indic
        Would be cleaner to calculate them from scratch (ie from all answer files)
        and to save them in a separate csv
        """
        df = self.df_per_indic.drop(
            [
                "indicator",
                "precision_vs_pdf",
                "recall_vs_pdf",
                "precision_vs_sispea",
                "recall_vs_sispea",
            ],
            axis=1,
        )
        # self.df_per_pdf could be also used but self.df_per_indic
        # has the advantage of having a column pdf_list_file

        # Compute the mean metrics
        print(f"Warning!!! Null values are ignored when computing the mean accuracy.")
        mean_df = df.groupby(
            ["benchmark_version", "pdf_list_file"], as_index=False
        ).mean()

        # ... Express the number of Null and the number of TP, TN, FP1, FP2, FN as rates
        tmp_df = mean_df.iloc[:, np.r_[2, 4:10, 11:16]]
        # Note that tmp_df.iloc[:, -5:].sum(axis=1) is the number of PDFs in each pdf_list_file for a given benchmark
        mean_df.iloc[:, np.r_[2, 4:10, 11:16]] = tmp_df.div(
            tmp_df.iloc[:, -5:].sum(axis=1), axis=0
        )

        # Add average precision and recall
        for ref in ["sispea", "pdf"]:
            mean_df[f"avg_precision_vs_{ref}"] = mean_df[f"tp_nb_vs_{ref}"] / (
                mean_df[f"tp_nb_vs_{ref}"]
                + mean_df[f"fp1_nb_vs_{ref}"]
                + mean_df[f"fp2_nb_vs_{ref}"]
            )
            mean_df[f"avg_recall_vs_{ref}"] = mean_df[f"tp_nb_vs_{ref}"] / (
                mean_df[f"tp_nb_vs_{ref}"]
                + mean_df[f"fn_nb_vs_{ref}"]
                + mean_df[f"fp2_nb_vs_{ref}"]
            )

        # Rename columns
        mean_df = mean_df.rename(
            {
                "no_value_indic_nb_in_sispea": "no_value_indic_rate_in_sispea",
                "accuracy_vs_sispea": "avg_accuracy_vs_sispea",
                "tp_nb_vs_sispea": "tp_rate_vs_sispea",
                "tn_nb_vs_sispea": "tn_rate_vs_sispea",
                "fp1_nb_vs_sispea": "fp1_rate_vs_sispea",
                "fp2_nb_vs_sispea": "fp2_rate_vs_sispea",
                "fn_nb_vs_sispea": "fn_rate_vs_sispea",
                "no_value_indic_nb_in_pdf": "no_value_indic_rate_in_pdf",
                "accuracy_vs_pdf": "avg_accuracy_vs_pdf",
                "tp_nb_vs_pdf": "tp_rate_vs_pdf",
                "tn_nb_vs_pdf": "tn_rate_vs_pdf",
                "fp1_nb_vs_pdf": "fp1_rate_vs_pdf",
                "fp2_nb_vs_pdf": "fp2_rate_vs_pdf",
                "fn_nb_vs_pdf": "fn_rate_vs_pdf",
            },
            axis=1,
        )

        return mean_df

    def save_metrics(self):
        df = self.df_per_pdf.sort_values(by=["benchmark_version", "pdf_name"])
        self.fs.write_df_to_csv(df, self.path_per_pdf, index=False)
        df = self.df_per_indic
        self.fs.write_df_to_csv(df, self.path_per_indic, index=False)
        print(f"Metrics per pdf are saved to {self.path_per_pdf.split("/")[-1]}")
        print(
            f"Metrics per indicator are saved to {self.path_per_indic.split("/")[-1]}"
        )
