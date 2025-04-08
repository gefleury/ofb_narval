import re

import pandas as pd
import pdfplumber
import PyPDF2
from unidecode import unidecode

from narval.utils import FileSystem

# Define the (local or S3) file system
fs = FileSystem()


class PyPDFReader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.textpages = []  # List to store the extracted text from each page

    def extract_raw_text(self, max_page=None):
        if max_page:
            assert isinstance(max_page, int)
        # Clear any existing text in self.text
        self.textpages.clear()
        # Loop through all the pages and extract text
        with fs.open(self.pdf_path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            for page in pdf.pages[:max_page]:
                page_text = page.extract_text()
                # Append the extracted text of each page to self.text
                self.textpages.append(page_text)


class PDFPlumberReader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        # List to store the extracted text from each page
        self.textpages = []
        # List to store the list of extracted dataframes from each page
        self.dftables = []

    def extract_raw_text(self):
        self.textpages.clear()
        with fs.open(self.pdf_path, "rb") as file:
            pdf = pdfplumber.open(file)
            for page in pdf.pages:
                page_text = page.extract_text()
                self.textpages.append(page_text)

    def extract_raw_tables(self):
        self.dftables.clear()
        with fs.open(self.pdf_path, "rb") as file:
            pdf = pdfplumber.open(file)
            for page in pdf.pages:
                df_list = []
                tabs = page.extract_tables(
                    table_settings={
                        "snap_x_tolerance": 6  # to solve the problem of added columns
                    }
                )
                for table in tabs:
                    header, *data = table
                    df = pd.DataFrame(data, columns=header)
                    df_list.append(df)
                self.dftables.append(df_list)


class PDFReader:
    def __init__(
        self,
        pdf_path: str,
        extract_text=True,
        extract_tables=True,
        text_extraction_method="PyPDF2",
        table_extraction_method="PDFPlumber",
    ):
        self.pdf_path = pdf_path
        # List to store the extracted text from each page
        self.textpages = []  # List of strings
        # Automatically extract text when initialized
        if extract_text:
            self.extract_text(method=text_extraction_method)
        # List to store the list of extracted dataframes from each page
        self.dftables = []  # List of list of pandas dataframes
        # Automatically extract tables when initialized
        if extract_tables:
            self.extract_tables(method=table_extraction_method)

    def extract_raw_text(self, method="PyPDF2"):
        if method == "PyPDF2":
            reader = PyPDFReader(self.pdf_path)
        elif method == "PDFPlumber":
            reader = PDFPlumberReader(self.pdf_path)
        else:
            raise ValueError(f"Unknown PDF Reader {method} for text extraction. ")

        reader.extract_raw_text()
        self.textpages = reader.textpages

    def post_process_extracted_text(self):
        # Replace unrecognized fonts
        # Temporary fix, to be improved with a better text extraction package
        def substitute_font(text):
            # eg in "Autorisa\x1aon"
            text = text.replace("\x1a", "ti")
            text = text.replace("\x1b", "ti")
            text = text.replace("(cid:27)", "ti")
            # superscript m3
            text = text.replace("m\u00b3", "m3")
            return text

        self.textpages = list(map(substitute_font, self.textpages))

    def extract_text(self, method="PyPDF2"):
        self.extract_raw_text(method=method)
        self.post_process_extracted_text()

    def extract_raw_tables(self, method="PDFPlumber"):
        if method == "PDFPlumber":
            reader = PDFPlumberReader(self.pdf_path)
        else:
            raise ValueError(f"Unknown PDF Reader {method} for table extraction. ")

        reader.extract_raw_tables()
        self.dftables = reader.dftables

    def clean_tables(self):
        def clean_df(df):
            # Rename columns named None
            df = df.rename(lambda x: x if x not in [None, ""] else "Col", axis=1)
            # Rename duplicate columns
            df_columns = df.columns.tolist()
            df.columns = [
                col if df_columns.count(col) == 1 else str(col) + "_" + str(i)
                for (i, col) in enumerate(df_columns)
            ]
            # Remove rows and columns with only NaN
            df = df.replace(to_replace="", value=None)
            df = df.dropna(axis="columns", how="all")
            df = df.dropna(axis="rows", how="all")
            # Clean text in each cell (remove extra spaces, new lines, ...)
            df = df.map(lambda x: " ".join(str(x).split()) if x is not None else x)

            return df

        self.dftables = [[clean_df(df) for df in sublist] for sublist in self.dftables]

    def extract_tables(self, method="PDFPlumber"):
        self.extract_raw_tables(method=method)
        self.clean_tables()

    def get_text(self):
        return self.textpages

    def is_toc_page_by_keywords(self, page_idx: int):
        """
        Determines whether or not self.textpages[page_idx] contains a table of contents
        using keywords

        Returns: boolean
        """
        keywords = ["Table des matières", "Sommaire"]
        text = self.textpages[page_idx]
        formatted_text = unidecode(text).lower()
        formatted_keywords = [unidecode(kw).lower() for kw in keywords]
        is_toc = any(kw in formatted_text for kw in formatted_keywords)

        return is_toc

    def is_toc_page_by_structure(self, page_idx: int, with_dots=True, min_matches=5):
        """
        Determines whether or not self.textpages[page_idx] contains a table of contents
        using the page structure

        Returns: boolean
        """
        text = self.textpages[page_idx]

        if with_dots:
            # Regex pattern to match lines like "Caractérisation ..... 12" (with dots in between)
            pattern = r".+?\.{4,}\s+\d{1,3}\s*"  # r".+\.{4,}\s+\d{1,3}\s*\n"
        else:
            # Regex pattern to match lines like "Caractérisation  12" (without dots in between)
            # or conversely "12 Caractérisation"
            # Dangerous because this tends to capture tables
            pattern = r"(.+\s*\d{1,3}\s*$|^\s*\d{1,3}\s+.+$)"

        matches = re.findall(pattern, unidecode(text), re.MULTILINE)

        # Page idx is a TOC page if there are multiple matches
        is_toc = len(matches) >= min_matches

        return is_toc

    def find_toc_pages(self, trunc_idx=10):
        """
        Find the pages containing the table of contents using keywords and page structure

        Parameters:
        trunc_idx : index of the last page considered for toc searching

        Returns:
        toc_pages_idx_list : a list of int ie the toc page indices (indices start from 0)
        """
        toc_pages_idx_list = []
        # Search the toc only in the "trunc" first pages
        idx_max = min(len(self.textpages) - 1, trunc_idx)
        for idx in range(idx_max + 1):
            is_toc_kw = self.is_toc_page_by_keywords(idx)
            is_toc_struct_1 = self.is_toc_page_by_structure(idx, with_dots=True)
            is_toc_struct_2 = self.is_toc_page_by_structure(idx, with_dots=False)
            # Toc page numbers found by 2 conditions : keywords + one of the 2 structures
            if is_toc_kw and (is_toc_struct_1 or is_toc_struct_2):
                toc_pages_idx_list.append(idx)
            # Add pages having the line structure with dots
            elif is_toc_struct_1:
                toc_pages_idx_list.append(idx)
            else:
                pass

        # For each identified toc page, check if the next page is also a toc page
        # using the condition on its structure with dots (stronger condition than the one without dots)
        toc_pages_idx_tmp = []
        for idx in toc_pages_idx_list:
            if self.is_toc_page_by_structure(idx + 1, with_dots=True, min_matches=3):
                toc_pages_idx_tmp.append(idx + 1)

        toc_pages_idx_list.extend(toc_pages_idx_tmp)

        return list(set(toc_pages_idx_list))

    def is_rad(self) -> bool:
        """
        Determines whether or not the PDF is a RAD
        by looking for keywords in the first page
        (imperfect, some RADs are not detected as RADs but there is no RPQS wrongly detected as RAD)

        Returns: boolean
        """
        keywords = ["rapport annuel du délégataire", "rapport annuel du prestataire"]
        text = self.textpages[0]
        formatted_text = unidecode(text).lower()
        formatted_keywords = [unidecode(kw).lower() for kw in keywords]
        is_rad = any(kw in formatted_text for kw in formatted_keywords)

        return is_rad
