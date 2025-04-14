import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from tqdm import tqdm
from unidecode import unidecode

from narval.utils import FileSystem, get_data_dir

FS = FileSystem()
DATA_DIR = get_data_dir()


class PDFScraperFromGoogle:
    path = Path(DATA_DIR + "/data/scraping/")

    def __init__(self, collectivity_filename: str, report_filename: str):
        self.rpqs_path = PDFScraperFromGoogle.path / Path("from_google")
        self.rpqs_path.mkdir(parents=True, exist_ok=True)
        self.collectivity_source = CollectivitySource(collectivity_filename)
        self.report = ScrapingReport(report_filename)

    @staticmethod
    def googlesearch(
        keyword_list: list[str], context_url=None, num_results=5
    ) -> list[str]:
        """
        Returns the first 'num_results' urls found on Goolge using keywords in keyword_list
        when the Google search is limited inside context_url (no limitation if context_url=None)
        """
        query = " ".join(keyword_list)
        # query = " ".join(['"' + kw + '"'  for kw in keywords])
        if context_url:
            query = f"{query} site:{context_url}"
        url_list = []
        try:
            for url in search(
                query,
                num_results=num_results,
                sleep_interval=5,
                lang="fr",
                region="fr",
                advanced=False,
                unique=True,
            ):
                url_list.append(url)
        except requests.exceptions.HTTPError:
            # Catch error 429 "Too many requests"
            # Wait before next use
            time.sleep(10)  # replace with a bool variable catching the error
            pass
        except Exception as e:
            print(f"Ignored error during google search: {e}")
            pass

        return url_list

    @staticmethod
    def build_keyword_list_list(
        collectivity_name: str, competence: str, year: str
    ) -> list[list[str]]:
        assert competence in valid_competence_list
        keyword_list_list = [
            ["RPQS", collectivity_name, competence, year],
            ["RAD", collectivity_name, competence, year],
            ["Rapport du maire", collectivity_name, competence, year],
        ]

        return keyword_list_list

    @staticmethod
    def url_to_filename(pdf_url: str) -> str:
        """
        Converts a PDF URL into a valid filename while keeping the full URL structure.
        """
        parsed_url = urlparse(pdf_url)
        full_path = (
            parsed_url.netloc + parsed_url.path + parsed_url.query + parsed_url.fragment
        )
        # Replace invalid filename characters with underscores
        filename = re.sub(r"[^a-zA-Z0-9_]", "_", full_path)
        filename = re.sub(r".pdf", "", filename)

        return filename

    def fetch_pdf_urls_for_one_entity(
        self, collectivity_name: str, competence: str, year: str, context_url=None
    ):
        kw_list_list = self.build_keyword_list_list(collectivity_name, competence, year)
        source_url_list_list = []
        pdf_url_list_list_list = []
        for keyword_list in kw_list_list:
            url_list = self.googlesearch(keyword_list, context_url=context_url)
            pdf_url_list_list = []
            noerror_url_list = []
            for url in url_list:
                extractor = PDFLinkExtractor(url)
                try:
                    pdf_url_list = extractor.fetch_pdf_urls()
                except requests.RequestException as e:
                    # Should investigate how to bypass this error
                    print(f"An error occured : {e}")
                    print(f"PDF reports are not searched in {url}")
                else:
                    noerror_url_list.append(url)
                    pdf_url_list_list.append(pdf_url_list)
            source_url_list_list.append(noerror_url_list)
            pdf_url_list_list_list.append(pdf_url_list_list)

        return kw_list_list, source_url_list_list, pdf_url_list_list_list

    def download_one_pdf(
        self, url: str, save_subfolder: str, save_filename: str
    ) -> Path:
        folder_path = self.rpqs_path / Path(save_subfolder)
        folder_path.mkdir(parents=True, exist_ok=True)
        save_path = folder_path / Path(save_filename)

        download_file(url, save_path)

        return save_path

    def download_all_pdfs(
        self, year=None, competence=None, status_lot=None, use_sispea_context_url=False
    ) -> None:
        # Extract the collectivity df
        df = self.collectivity_source.filter_collectivity_df_for_googlescraper(
            year=year, competence=competence, status_lot=status_lot
        )

        print("Downloading RPQS ...")
        # Loop over sispea entities
        for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
            year = row.year
            zip_code = row.zip_code
            municipality_name = unidecode(row.linked_municipality_name.replace(" ", ""))
            competence_tag = competence_dict[competence]  # EP, AC, ANC
            if use_sispea_context_url:
                # Check for NaN
                context_url = (
                    row.link if (row.link and not row.link == row.link) else None
                )
            else:
                context_url = None
            save_subfolder = f"{year}/{competence_tag}"
            save_filename_front = (
                f"RPQS_{municipality_name}_cp{zip_code}_{competence_tag}_{year}"
            )
            # Get PDF urls for this entity
            kw_ll, src_url_ll, pdf_url_lll = self.fetch_pdf_urls_for_one_entity(
                municipality_name, competence, year, context_url=context_url
            )
            # Loop over url
            for kw_l, src_url_l, pdf_url_ll in zip(kw_ll, src_url_ll, pdf_url_lll):
                for src_url, pdf_url_l in zip(src_url_l, pdf_url_ll):
                    for pdf_url in pdf_url_l:
                        save_filename = (
                            f"{save_filename_front}_{self.url_to_filename(pdf_url)}.pdf"
                        )
                        try:
                            # Download the RPQS
                            outpath = self.download_one_pdf(
                                pdf_url, save_subfolder, save_filename
                            )
                        except FileExistsError:
                            print(f"The file {save_filename} already exists.")
                        else:
                            if not outpath.is_relative_to(Path(DATA_DIR)):
                                raise ValueError(f"The path {outpath} is ill-defined.")
                            # Update the scraping report
                            row_data = {col: getattr(row, col) for col in df.columns}
                            row_data.update(
                                {
                                    "context_link_if_from_sispea": context_url,
                                    "keywords_if_from_google": kw_l,
                                    "source_url_if_from_google": src_url,
                                    "file_path": outpath.relative_to(Path(DATA_DIR)),
                                    "file_tag": "pdf",
                                    "scraping_date": pd.to_datetime("today").strftime(
                                        "%d/%m/%Y"
                                    ),
                                }
                            )
                            self.report.update_report(row_data)


class PDFScraperFromSispea:
    path = Path(DATA_DIR + "/data/scraping/")

    def __init__(self, collectivity_filename: str, report_filename: str):
        self.rpqs_path = PDFScraperFromSispea.path / Path("from_sispea")
        self.rpqs_path.mkdir(parents=True, exist_ok=True)
        self.collectivity_source = CollectivitySource(collectivity_filename)
        self.report = ScrapingReport(report_filename)

    @staticmethod
    def build_sispea_rpqs_url(collectivity_id: str, rpqs_id: str):
        base_url = (
            "https://www.services.eaufrance.fr/sispea//referential/download-rpqs.action"
        )
        rpqs_url = (
            f"{base_url}?collectivityId={str(collectivity_id)}&rpqsId={str(rpqs_id)}"
        )
        return rpqs_url

    def download_one_rpqs_from_ids(
        self,
        collectivity_id: str,
        rpqs_id: str,
        save_subfolder: str,
        save_filename: str,
    ) -> Path:
        folder_path = self.rpqs_path / Path(save_subfolder)
        folder_path.mkdir(parents=True, exist_ok=True)
        save_path = folder_path / Path(save_filename)

        rpqs_url = self.build_sispea_rpqs_url(collectivity_id, rpqs_id)
        download_file(rpqs_url, save_path)

        return save_path

    def download_all_rpqs_from_ids(
        self, year=None, competence=None, status_lot=None
    ) -> None:
        # Extract the collectivity df
        df = self.collectivity_source.filter_collectivity_df_for_sispeascraper(
            year=year, competence=competence, status_lot=status_lot, rpqs_type="File"
        )

        print("Downloading RPQS ...")
        # Loop over rpqs
        for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
            year = row.year
            collectivity_id = row.id_collectivity
            rpqs_id = row.id_rpqs
            filetag = row.file_name.split(".")[-1].lower()  # eg .pdf
            zip_code = row.zip_code
            municipality_name = unidecode(row.linked_municipality_name.replace(" ", ""))
            competence_tag = competence_dict[competence]  # EP, AC, ANC
            save_subfolder = f"{year}/{competence_tag}"
            save_filename = f"RPQS_{municipality_name}_cp{zip_code}_rpqsid_{rpqs_id}_{competence_tag}_{year}.{filetag}"

            try:
                # Download the RPQS
                full_filepath = self.download_one_rpqs_from_ids(
                    collectivity_id, rpqs_id, save_subfolder, save_filename
                )
            except FileExistsError:
                print(f"The file {save_filename} already exists.")
            else:
                if not full_filepath.is_relative_to(Path(DATA_DIR)):
                    raise ValueError(f"The path {full_filepath} is ill-defined.")
                # Update the scraping report
                row_data = {col: getattr(row, col) for col in df.columns}
                row_data.update(
                    {
                        "id_rpqs_if_from_sispea": rpqs_id,
                        "rpqs_type_if_from_sispea": row.rpqs_type,
                        "sispea_file_name_if_from_sispea": row.file_name,
                        "file_path": full_filepath.relative_to(Path(DATA_DIR)),
                        "file_tag": filetag,
                        "scraping_date": pd.to_datetime("today").strftime("%d/%m/%Y"),
                    }
                )
                self.report.update_report(row_data)
                # If the downloaded file is a zip file,
                # unzip it and update the scraping report
                if filetag == "zip":
                    self.unzip_file_and_update_report(row_data)

        return None

    def unzip_file_and_update_report(self, file_data: dict):
        """
        Unzip file and move+rename the files inside to the parent directory
        if the original file name does not contain "delib" or "délib".
        The scraping report is finally updated.

        Parameters:
            file_data (dict): File metadata as calculated in download_all_rpqs_from_ids
        """
        zip_filepath = Path(DATA_DIR) / Path(file_data["file_path"])
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            extract_folder = zip_filepath.with_name(zip_filepath.stem)
            zip_ref.extractall(extract_folder)
        for file in extract_folder.iterdir():
            i = 0
            if file.is_file() and not re.search(r"delib|délib", file.name):
                i = i + 1  # count the different files in the unzipped folder
                filepath = zip_filepath.with_name(
                    f"{zip_filepath.stem}_{i}{file.suffix}"
                )
                file.rename(filepath)
                file_data.update(
                    {
                        "file_path": filepath.relative_to(Path(DATA_DIR)),
                        "file_tag": filepath.suffix[1:],  # eg pdf
                    }
                )
                self.report.update_report(file_data)
        extract_folder.rmdir()

    # @staticmethod
    # def convert_to_pdf(input_path: Path) -> Path:
    #     suffix = input_path.suffix
    #     assert suffix in [".doc", ".docx", ".rtf"]
    #     output_path = input_path.with_suffix(".pdf")
    #     pypandoc.convert_file(input_path, "pdf", outputfile=output_path)

    #     return output_path


class CollectivitySource:
    path = Path(DATA_DIR + "/data/scraping/")

    def __init__(self, collectivity_filename: str):
        self.collectivity_filepath = CollectivitySource.path / Path(
            collectivity_filename
        )

    def filter_collectivity_df(
        self, year=None, competence=None, status_lot=None
    ) -> pd.DataFrame:
        if competence and competence not in valid_competence_list:
            raise ValueError(
                f"Unknown competence {competence}. Valid competences are : {valid_competence_list}"
            )

        if status_lot and status_lot not in valid_status_lot_list:
            raise ValueError(
                f"Unknown SISPEA status {status_lot}. Valid status are : {valid_status_lot_list}"
            )

        cols_to_keep = [
            "year",
            "collectivity_name",
            "linked_municipality_name",
            "id_linked_municipality",
            "zip_code",
            "id_collectivity",
            "id_service",
            "id_rpqs",
            "file_name",
            "link",
            "name_competence",
            "status_lot",
            "rpqs_type",
        ]
        df = FS.read_csv_to_df(
            self.collectivity_filepath,
            usecols=cols_to_keep,
            engine="python",
        )

        if year:
            year = int(year)
            df = df.query("year==@year")
        if competence:
            df = df.query("name_competence==@competence")
        if status_lot:
            df = df.query("status_lot==@status_lot")
        df = df.drop_duplicates()

        # Convert ids to str
        for col in df.select_dtypes("number"):
            df[col] = df[col].apply(lambda x: str(int(x)) if not pd.isna(x) else x)

        return df

    def filter_collectivity_df_for_sispeascraper(
        self, year=None, competence=None, status_lot=None, rpqs_type="File"
    ) -> pd.DataFrame:
        if rpqs_type not in valid_rpqs_type_list:
            raise ValueError(
                f"Unknown rpqs type {rpqs_type}. Valid rpqs types are : {valid_rpqs_type_list}"
            )
        df = self.filter_collectivity_df(
            year=year, competence=competence, status_lot=status_lot
        )
        df = df.query("rpqs_type==@rpqs_type")

        if rpqs_type == "File":
            df = df.query(  # Remove deliberation reports
                "file_name.str.contains(r'delib|délib', case=False, regex=True)==False"
            )

        df = df.drop("link", axis="columns").drop_duplicates()

        return df

    def filter_collectivity_df_for_googlescraper(
        self, year=None, competence=None, status_lot=None
    ):
        df = self.filter_collectivity_df(
            year=year, competence=competence, status_lot=status_lot
        )
        # Exclude cases with rpqs_type=File as they are scrapped from SISPEA
        df = df.query("rpqs_type!='File'")
        df = df.drop(["id_rpqs", "file_name", "rpqs_type"], axis="columns")
        df = df.drop_duplicates()

        return df


class ScrapingReport:
    folder_path = Path(DATA_DIR + "/data/scraping/")

    def __init__(self, report_name: str):
        self.name = report_name
        ScrapingReport.folder_path.mkdir(parents=True, exist_ok=True)
        self.report_path = ScrapingReport.folder_path / Path(self.name)
        self.df = None

    def init_report_df(self):
        try:
            self.df = FS.read_csv_to_df(self.report_path)
        except FileNotFoundError:
            columns = [
                "year",
                "id_collectivity",
                "collectivity_name",
                "linked_municipality_name",
                "zip_code",
                "id_linked_municipality",
                "name_competence",
                "id_service",
                "id_rpqs_if_from_sispea",
                "context_link_if_from_sispea",
                "rpqs_type_if_from_sispea",
                "sispea_file_name_if_from_sispea",
                "keywords_if_from_google",
                "source_url_if_from_google",
                "status_lot",
                "file_path",
                "file_tag",
                "scraping_date",
            ]
            self.df = pd.DataFrame(columns=columns)

    def fill_report_df(self, row_data: dict) -> None:
        new_row = pd.DataFrame([row_data], columns=self.df.columns)
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def save_report(self) -> None:
        assert self.df.astype(str).duplicated().sum() == 0
        FS.write_df_to_csv(self.df, self.report_path, index=False)

    def update_report(self, row_data: dict) -> None:
        self.init_report_df()
        self.fill_report_df(row_data)
        self.save_report()


class PDFLinkExtractor:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def clean_pdf_url(self, pdf_link: str) -> str:
        """
        Cleans and normalizes a given PDF URL.

        - Converts relative URLs to absolute URLs using self.base_url.
        - Ensures proper schemes (http/https).
        - Filters out non-http/https links (like 'file://', 'ftp://').

        Parameters:
            pdf_link (str): The raw PDF link (extracted from the <a> tag).

        Returns:
            str: The cleaned absolute PDF URL or an empty string if invalid.
        """
        if not pdf_link:
            return ""  # Invalid URL

        # If the link is already absolute with http or https, return as is
        parsed_link = urlparse(pdf_link)
        if parsed_link.scheme in ["http", "https"]:
            return pdf_link

        # If the link starts with "//", it’s missing a scheme (e.g., //example.com/file.pdf)
        if pdf_link.startswith("//"):
            return "https:" + pdf_link  # Default to HTTPS

        # If it's a relative URL (no scheme or netloc), resolve it using self.base_url
        cleaned_url = urljoin(self.base_url, pdf_link)

        # Ensure the final URL is http or https
        if not is_pdf_url(cleaned_url):
            return ""  # Invalid URL

        return cleaned_url

    def fetch_pdf_urls(self) -> list[str]:
        """Extract all PDF links from the base URL."""
        if is_pdf_url(self.base_url):
            return [self.base_url]

        try:
            response = requests.get(self.base_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Request error from url {self.base_url}"
            ) from e

        html_content = None
        if response and response.ok:  # Check if response is valid
            html_content = response.text
        if not html_content:
            raise ValueError(
                f"Error: Failed to fetch HTML content from url {self.base_url}"
            )

        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception as e:
            raise Exception(
                f"Could not parse HTML with BeautifulSoup from url {self.base_url}"
            ) from e

        # Find all hyperlinks present on webpage
        links = soup.find_all("a", href=True)
        # From all links check for pdf link and if present, fetch it
        url_set = set()
        for link in links:
            pdf_link = link.get("href")
            ###################### To be ckecked/completed ############################
            # There is no need to restrict to URL ending with .pdf (can be truc.pdf?truc=01)
            if pdf_link and pdf_link.lower().endswith(".pdf"):
                pdf_link = self.clean_pdf_url(pdf_link)
                if pdf_link:  # Ensure it's a valid cleaned URL
                    url_set.add(pdf_link)

        return list(url_set)


#######################


def download_file(url: str, save_path: Path) -> None:
    """
    Downloads a file from a given URL and saves it to the specified filepath.

    Parameters:
        url (str): The URL of the file.
        save_path (Path): Path to the output path

    Raises:
        requests.RequestException : in case of requests error
        ValueError: If the file cannot be downloaded.
        FileExistsError: if the filename already exists
        Exception: If there is an issue writing the file.
    """
    if save_path.exists():
        raise FileExistsError(f"The file {save_path.name} already exists.")

    # Fetch file content
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()  # Raise HTTP Error
    except requests.RequestException as e:
        raise requests.RequestException(f"Request error from url {url}") from e

    file_content = None
    if response and response.ok:  # Check if response is valid
        file_content = response.content
    if file_content is None:
        raise ValueError(
            f"Error: Failed to fetch file content from url {url} (Status: {response.status_code})"
        )

    # Save file
    try:
        save_path.write_bytes(file_content)
    except Exception as e:
        raise Exception(f"Error saving file from {url}") from e

    return None


def is_pdf_url(url: str) -> bool:
    try:
        r = requests.head(url, timeout=10)
        condition_1 = "application/pdf" in r.headers.get("content-type", "")
        condition_2 = urlparse(url).scheme in ["http", "https"]
        return condition_1 and condition_2
    except requests.RequestException:
        return False


####################

competence_dict = {
    "eau potable": "EP",
    "assainissement collectif": "AC",
    "assainissement non collectif": "ANC",
}

valid_competence_list = [
    "eau potable",
    "assainissement collectif",
    "assainissement non collectif",
]

valid_status_lot_list = [
    "InputInProgress",
    "Published",
    "PublishedWithoutVerification",
    "VerificationInProgress",
    "Verified",
    "WaitingForInput",
    "WaitingForVerification",
]

valid_rpqs_type_list = ["File", "Url"]
