import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    ENV_FILE_PATH = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)
except ImportError:
    pass

DATASET_ID = "Tobi-Bueck/customer-support-tickets"

TEV_MODEL_NAME = "together/Tev1-4B-experimental"
JEV_MODEL_NAME = "jev-latest"
# Local Core ML port of the English Laya checkpoint (convaiinnovations/laya).
LAYA_MODEL_NAME = os.getenv("LAYA_MODEL", "aac6fef/laya-coreml").strip() or "aac6fef/laya-coreml"
OPENAI_MODEL_NAME = "gpt-4o"

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"

# USD per 1,000,000 tokens. Decision APIs bill input tokens; output is free.
# Laya Core ML runs on this machine, so it has no token price.
TEV_INPUT_PRICE_PER_M = 0.042
TEV_OUTPUT_PRICE_PER_M = 0.0
JEV_INPUT_PRICE_PER_M = 0.042
JEV_OUTPUT_PRICE_PER_M = 0.0
GPT4O_INPUT_PRICE_PER_M = 2.50
GPT4O_OUTPUT_PRICE_PER_M = 10.00


def token_cost(input_tokens: int, output_tokens: int, input_price: float, output_price: float) -> float:
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def env_key(*names: str) -> str:
    placeholders = {"", "your_together_api_key_here", "your_openai_api_key_here", "your_typesafe_api_key_here"}
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        stripped = value.strip()
        if stripped in placeholders or stripped.startswith("your_"):
            continue
        return stripped
    return ""
