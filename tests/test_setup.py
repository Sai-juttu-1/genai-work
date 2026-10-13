"""Session 2 checkpoint tests."""

import os
import sys

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)


def test_python_version():
    assert sys.version_info[:2] == (
        3,
        12,
    ), "Run commands with `uv run`, never plain `python`."


def test_env_present():
    required_keys = ("BASE_URL", "API_KEY", "MODEL")

    if not all(os.getenv(key) for key in required_keys):
        pytest.skip("API environment variables are not configured in CI.")

    assert "paste_your" not in os.getenv("API_KEY")
    assert not os.getenv("API_KEY").startswith(("'", '"'))


def test_model_answers():
    required_keys = ("BASE_URL", "API_KEY", "MODEL")

    if not all(os.getenv(key) for key in required_keys):
        pytest.skip("API environment variables are not configured in CI.")

    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
    )

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with OK"}],
    )

    assert response.choices[0].message.content
