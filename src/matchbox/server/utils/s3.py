from typing import BinaryIO

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError


def upload_to_s3(
    file: BinaryIO, bucket_name: str, object_name: str | None = None
) -> bool:
    """
    Upload a file to an S3 bucket.

    Args:
        file (BinaryIO): File to upload.
        bucket_name (str): Target S3 bucket.
        object_name (str): S3 object name. If not specified, file_name is used.

    Returns:
        bool: True if the file was uploaded, else False.
    """
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_fileobj(file, bucket_name, object_name)
        return True
    except NoCredentialsError:
        return False
    except PartialCredentialsError:
        return False
