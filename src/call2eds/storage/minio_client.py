import hashlib
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config

from call2eds.config.settings import settings


class MinioClient:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=f"http{'s' if settings.minio_secure else ''}://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.bucket = settings.minio_bucket

    def ensure_bucket(self):
        buckets = [b["Name"] for b in self.client.list_buckets().get("Buckets", [])]
        if self.bucket not in buckets:
            self.client.create_bucket(Bucket=self.bucket)

    def upload_file(self, local_path: Path, key: str) -> str:
        self.ensure_bucket()
        self.client.upload_file(str(local_path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download_file(self, key: str, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


def get_minio() -> MinioClient:
    return MinioClient()
