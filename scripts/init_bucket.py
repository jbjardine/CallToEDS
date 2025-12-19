#!/usr/bin/env python3
"""Crée le bucket MinIO si absent."""
from call2eds.storage.minio_client import get_minio


def main():
    client = get_minio()
    client.ensure_bucket()
    print(f"Bucket '{client.bucket}' prêt")



if __name__ == "__main__":
    main()
