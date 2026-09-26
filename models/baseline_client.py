import os
import time
import re
from typing import Optional
from dataclasses import dataclass
from openai import OpenAI

from config import (
    BASELINE_MODEL_NAME,
    TAXONOMY_OPTIONS,
    OPTION_LETTER_TO_NAME,
    GPT4O_INPUT_PRICE_PER_M,
    GPT4O_OUTPUT_PRICE_PER_M
)

@dataclass
class BaselineClassificationResult:
    predicted_letter: str
    predicted_category: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    raw_response: str
    parse_success: bool
    cost_usd: float

@dataclass
class DownstreamResult:
    content: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float

class BaselineClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key or self.api_key.strip() in {"your_openai_api_key_here", ""} or self.api_key.startswith("your_"):
            raise ValueError(
                "Invalid or missing OPENAI_API_KEY in your .env file.\n"
                "Please open '/Users/vanshpruthi/tev-playground/.env' and replace 'your_openai_api_key_here' with your real OpenAI API key."
            )
        self.client = OpenAI(api_key=self.api_key.strip())
        self.model = BASELINE_MODEL_NAME

    def classify(self, text: str) -> BaselineClassificationResult:
        """
        Executes classification using monolithic baseline model (GPT-4o).
        Instructs model to return only the single category letter.
        """
        options_formatted = "\n".join(TAXONOMY_OPTIONS)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict text classification system. Classify the given news article "
                    "into exactly one of the following categories:\n"
                    f"{options_formatted}\n\n"
                    "Respond with ONLY the single letter (A, B, C, or D). Do not include any other explanation."
                )
            },
            {
                "role": "user",
                "content": f"Article:\n{text}"
            }
        ]

        start_time = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=8
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        raw_content = response.choices[0].message.content.strip()

        match = re.search(r"\b([A-D])\b", raw_content.upper())
        if match:
            letter = match.group(1)
            parse_success = True
            category = OPTION_LETTER_TO_NAME.get(letter, "Unknown")
        else:
            letter = raw_content[:1].upper() if raw_content and raw_content[0].upper() in "ABCD" else "Unknown"
            parse_success = letter != "Unknown"
            category = OPTION_LETTER_TO_NAME.get(letter, "Unknown")

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        cost_usd = (input_tokens * (GPT4O_INPUT_PRICE_PER_M / 1_000_000)) + (output_tokens * (GPT4O_OUTPUT_PRICE_PER_M / 1_000_000))

        return BaselineClassificationResult(
            predicted_letter=letter,
            predicted_category=category,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=raw_content,
            parse_success=parse_success,
            cost_usd=cost_usd
        )

    def generate_downstream(self, text: str, category: str) -> DownstreamResult:
        """
        Executes selective downstream generation using GPT-4o.
        Generates executive briefing and action points for actionable categories (Business, Sci/Tech).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an enterprise analyst agent. Provide an executive summary and key action "
                    "points for the following news item. Keep your response concise (2-3 bullet points)."
                )
            },
            {
                "role": "user",
                "content": f"Category: {category}\n\nArticle: {text}\n\nKey takeaways and recommended action:"
            }
        ]

        start_time = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=150
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        cost_usd = (input_tokens * (GPT4O_INPUT_PRICE_PER_M / 1_000_000)) + (output_tokens * (GPT4O_OUTPUT_PRICE_PER_M / 1_000_000))

        return DownstreamResult(
            content=content,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd
        )
