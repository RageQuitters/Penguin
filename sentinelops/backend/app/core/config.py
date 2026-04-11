"""
Core configuration — reads from environment variables.
All Huawei Cloud service endpoints are configured here.
Swap .env values to point at real Huawei Cloud services in production.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "SentinelOps"
    app_version: str = "1.0.0"
    debug: bool = False

    # -----------------------------------------------------------------------
    # Huawei Cloud — Identity & Access Management (IAM)
    # -----------------------------------------------------------------------
    huawei_region: str = os.getenv("HUAWEI_REGION", "ap-southeast-3")  # Singapore
    huawei_project_id: str = os.getenv("HUAWEI_PROJECT_ID", "")
    huawei_iam_username: str = os.getenv("HUAWEI_IAM_USERNAME", "")
    huawei_iam_password: str = os.getenv("HUAWEI_IAM_PASSWORD", "")
    huawei_iam_domain: str = os.getenv("HUAWEI_IAM_DOMAIN", "")

    # -----------------------------------------------------------------------
    # Huawei Cloud — Pangu LLM (L3 Orchestrator)
    # ModelArts inference endpoint or Pangu API Gateway
    # -----------------------------------------------------------------------
    pangu_api_base: str = os.getenv("PANGU_API_BASE", "")
    pangu_api_key: str = os.getenv("PANGU_API_KEY", "")
    pangu_model: str = os.getenv("PANGU_MODEL", "pangu-chat")
    pangu_auth_mode: str = os.getenv("PANGU_AUTH_MODE", "apikey")  # "apikey" | "iam"

    # -----------------------------------------------------------------------
    # Huawei Cloud — ModelArts (L3 ML inference endpoints)
    # -----------------------------------------------------------------------
    modelarts_urgency_endpoint: str = os.getenv("MODELARTS_URGENCY_ENDPOINT", "")
    modelarts_fault_endpoint: str = os.getenv("MODELARTS_FAULT_ENDPOINT", "")
    modelarts_api_key: str = os.getenv("MODELARTS_API_KEY", "")

    # -----------------------------------------------------------------------
    # Huawei Cloud — GaussDB (L2 operational database)
    # -----------------------------------------------------------------------
    gaussdb_host: str = os.getenv("GAUSSDB_HOST", "")
    gaussdb_port: int = int(os.getenv("GAUSSDB_PORT", "5432"))
    gaussdb_name: str = os.getenv("GAUSSDB_NAME", "sentinelops")
    gaussdb_user: str = os.getenv("GAUSSDB_USER", "")
    gaussdb_password: str = os.getenv("GAUSSDB_PASSWORD", "")

    # -----------------------------------------------------------------------
    # Huawei Cloud — OBS (L2 object storage)
    # -----------------------------------------------------------------------
    obs_endpoint: str = os.getenv("OBS_ENDPOINT", "")
    obs_bucket: str = os.getenv("OBS_BUCKET", "sentinelops-models")
    obs_access_key: str = os.getenv("OBS_ACCESS_KEY", "")
    obs_secret_key: str = os.getenv("OBS_SECRET_KEY", "")

    # -----------------------------------------------------------------------
    # Huawei Cloud — DMS for Kafka (L1 streaming bus)
    # -----------------------------------------------------------------------
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "")
    kafka_topic_raw: str = os.getenv("KAFKA_TOPIC_RAW", "raw-readings")
    kafka_topic_processed: str = os.getenv("KAFKA_TOPIC_PROCESSED", "processed-readings")
    kafka_username: str = os.getenv("KAFKA_USERNAME", "")
    kafka_password: str = os.getenv("KAFKA_PASSWORD", "")

    # -----------------------------------------------------------------------
    # Huawei Cloud — SMSGMS / Telegram (L4 alerting)
    # -----------------------------------------------------------------------
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    smsmsg_endpoint: str = os.getenv("SMSMSG_ENDPOINT", "")
    smsmsg_app_key: str = os.getenv("SMSMSG_APP_KEY", "")
    smsmsg_app_secret: str = os.getenv("SMSMSG_APP_SECRET", "")

    # -----------------------------------------------------------------------
    # Local model paths (fallback when ModelArts endpoints not configured)
    # -----------------------------------------------------------------------
    urgency_model_path: str = "joblib_files/urgency_model.joblib"
    fault_model_path: str = "joblib_files/fault_model.joblib"
    dataset_path: str = "dataset/train_test.csv"

    feature_labels: list[str] = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    fault_labels: list[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
