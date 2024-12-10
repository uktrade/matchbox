import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def upload_to_s3(file, bucket_name, object_name=None):
    """
    Upload a file to an S3 bucket.

    :param file: File to upload
    :param bucket_name: Target S3 bucket
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if the file was uploaded, else False
    """
    # Initialize the S3 client
    s3_client = boto3.client('s3')
    try:
        # Upload the file
        s3_client.upload_fileobj(file, bucket_name, object_name)
        return True
    except NoCredentialsError:
        return False
    except PartialCredentialsError:
        return False


