import os
import sys

import pandas as pd
import s3fs


# Returns the path to the git repo
def get_git_root(current_dir=None):
    # If no directory is provided, use the current working directory
    if current_dir is None:
        current_dir = os.getcwd()

    # Traverse upwards until we find a .git folder or reach the root directory
    while current_dir != os.path.dirname(current_dir):
        # Keep going up until we reach the root "/"
        if os.path.isdir(os.path.join(current_dir, ".git")):
            return current_dir
        # Move one level up in the directory tree
        current_dir = os.path.dirname(current_dir)

    # If we reach here, we didn't find a .git directory
    return None


# Returns the path to the directory where the folder data is located
def get_data_dir():
    try:
        # directory where the folder data is located on the s3 bucket
        s3_bucket = os.environ["S3_BUCKET"]
        data_dir = "narval"
        return s3_bucket + "/" + data_dir
    except KeyError:
        # local directory where the folder data is located
        return get_git_root()
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


class FileSystem:
    def __init__(self):
        """
        Initialize the FileSystem class by automatically detecting whether
        the file system is local or S3 based on the existence
        of the environment variable "AWS_S3_ENDPOINT"
        """
        try:
            # Initialize the S3FileSystem
            s3_endpoint_url = "https://" + os.environ["AWS_S3_ENDPOINT"]
            self.fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint_url})
        except KeyError:
            # No need to do anything for local paths, default mode will be local
            self.fs = None
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def open(self, path, mode="r", **kwargs):
        """
        Open a file either on the local file system or in an S3 bucket.

        :param mode: Mode for opening the file, e.g. 'r' for reading, 'w' for writing.
        :param path: Path to the file to be opened.
        :param kwargs: Additional arguments to pass to the underlying open functions.
        :return: A file-like object, either for local files or S3 files.
        """
        if self.fs is None:
            # Open local file
            return open(path, mode=mode, **kwargs)
        else:
            # Open S3 file
            return self.fs.open(path, mode=mode, **kwargs)

    def read_csv_to_df(self, path, **kwargs):
        """
        Read a CSV file either from the local file system or from an S3 bucket.

        :param path: Path to the file to be read.
        :param kwargs: Additional arguments to pass to pandas read_csv function.
        :return: DataFrame containing the CSV data.
        """
        with self.open(path, mode="r") as file_in:
            df = pd.read_csv(file_in, **kwargs)
        return df

    def write_df_to_csv(self, df, path, **kwargs):
        """
        Write a DataFrame to a CSV file either to the local file system or to an S3 bucket.

        :param df: DataFrame to write.
        :param path: Path to the output file.
        :param kwargs: Additional arguments to pass to pandas to_csv function.
        """
        with self.open(path, mode="w", encoding="utf-8-sig") as file_out:
            df.to_csv(file_out, **kwargs)
