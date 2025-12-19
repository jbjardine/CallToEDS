import os
from functools import lru_cache


class Settings:
    def __init__(self):
        self.database_url = os.getenv(
            "DATABASE_URL", "postgresql+psycopg2://call2eds:call2eds@localhost:5432/call2eds"
        )
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        # endpoint externe pour les liens de téléchargement (ex: http://localhost:9000)
        self.minio_public_endpoint = os.getenv("MINIO_PUBLIC_ENDPOINT", f"http://{self.minio_endpoint}")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY", "call2eds")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY", "call2edssecret")
        self.minio_bucket = os.getenv("MINIO_BUCKET", "call2eds")
        self.minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        self.call2eds_model = os.getenv("CALL2EDS_MODEL", "small")
        self.call2eds_lang = os.getenv("CALL2EDS_LANG", "fr")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.web_host = os.getenv("CALL2EDS_WEB_HOST", "0.0.0.0")
        self.web_port = int(os.getenv("CALL2EDS_WEB_PORT", "8000"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
