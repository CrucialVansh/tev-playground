import os
from pathlib import Path
from typing import Dict, Set

# Explicitly load .env file from the project directory
try:
    from dotenv import load_dotenv
    ENV_FILE_PATH = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)
except ImportError:
    pass

# ---------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------
TEV_MODEL_NAME: str = "together/Tev1-4B-experimental"
BASELINE_MODEL_NAME: str = "gpt-4o"

# ---------------------------------------------------------
# Pricing Models (USD per 1,000,000 tokens)
# ---------------------------------------------------------
# Together AI Tev1-4B: $0.042 per million input tokens, output tokens are free
TEV_INPUT_PRICE_PER_M: float = 0.042
TEV_OUTPUT_PRICE_PER_M: float = 0.000

# OpenAI GPT-4o: $2.50 per million input tokens, $10.00 per million output tokens
GPT4O_INPUT_PRICE_PER_M: float = 2.500
GPT4O_OUTPUT_PRICE_PER_M: float = 10.000

# ---------------------------------------------------------
# AG News Dataset Taxonomy & Mappings
# ---------------------------------------------------------
# Class indices in AG News: 0: World, 1: Sports, 2: Business, 3: Sci/Tech
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

OPTION_LETTER_TO_NAME: Dict[str, str] = {
    "A": "World",
    "B": "Sports",
    "C": "Business",
    "D": "Sci/Tech"
}

NAME_TO_OPTION_LETTER: Dict[str, str] = {
    "World": "A",
    "Sports": "B",
    "Business": "C",
    "Sci/Tech": "D"
}

# The option strings formatted for classification prompts
TAXONOMY_OPTIONS = [
    "A: World",
    "B: Sports",
    "C: Business",
    "D: Sci/Tech"
]

# ---------------------------------------------------------
# Architectural Selective Downstream Policy
# ---------------------------------------------------------
# Only queries classified under these categories trigger downstream GPT-4o synthesis/action
# (e.g. In an enterprise system, business and tech alerts need automated briefing/processing)
ACTIONABLE_CATEGORIES: Set[str] = {"Business", "Sci/Tech"}
