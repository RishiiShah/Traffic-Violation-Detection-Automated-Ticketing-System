import os
import boto3

def upload_to_s3(file_path: str,
                 bucket_name: str,
                 region: str,
                 s3_prefix: str = "pdfs/") -> str:
    """
    Uploads `file_path` to S3, under `s3_prefix + basename`, and returns its HTTPS URL.
    Expects AWS creds & region either in env vars or passed here.
    """
    client = boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key = os.getenv("AWS_SECRET_KEY"),
        region_name           = region
    )
    key = os.path.join(s3_prefix, os.path.basename(file_path))
    client.upload_file(file_path, bucket_name, key)
    return f"https://{bucket_name}.s3.{region}.amazonaws.com/{key}"
