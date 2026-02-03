# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import base64
import os
from typing import Any, Dict, Optional

import requests

# vLLM API server configuration
API_BASE = "http://l20-2:9999/v1"
API_BASE = "http://l20-2:8000/v1"
MODEL_NAME = None  # Auto-detect if None


def get_model_name(api_base: str) -> str:
    """Get the first available model from the server."""
    response = requests.get(f"{api_base}/models")
    response.raise_for_status()
    models = response.json()
    return models["data"][0]["id"]


def load_audio_base64(audio_path: str) -> str:
    """Load audio file and encode to base64."""
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode("utf-8")


def call_audio_llm_api(
    audio_path: str,
    question_text: str,
    answer_text: str = "<think>\n",
    api_base: str = API_BASE,
    model_name: Optional[str] = MODEL_NAME,
    max_tokens: int = 16000,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    stop_token_ids: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Call the audio LLM API with the given audio and question.

    Args:
        audio_path: Path to the audio file
        question_text: The question/prompt text
        answer_text: Initial assistant response (default: "<think>\n" for R1 model)
        api_base: API server base URL
        model_name: Model name (auto-detect if None)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
        stop_token_ids: List of token IDs to stop generation

    Returns:
        The full API response as a dictionary
    """
    if stop_token_ids is None:
        stop_token_ids = [151665]

    # Auto-detect model if not specified
    model = model_name or get_model_name(api_base)

    # Validate audio file
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load and encode audio
    audio_base64 = load_audio_base64(audio_path)

    # Build payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_text},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }
                    }
                ]
            },
            {"role": "assistant", "content": answer_text},
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "stop_token_ids": stop_token_ids,
        "continue_final_message": True,
        "add_generation_prompt": False,
        "skip_special_tokens": False,
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()

    return response.json()


def extract_text_from_result(result: Dict[str, Any]) -> str:
    """
    Extract text from API result.

    Tries tts_content.tts_text first, falls back to content.
    (Aligned with stepaudior1vllm.py offline() method)

    Args:
        result: The API response dictionary

    Returns:
        The extracted text string
    """
    choice_data = result['choices'][0]
    message = choice_data['message']

    # Try tts_content.tts_text first, fallback to content
    text = message.get('tts_content', {}).get('tts_text', None)
    text = text if text is not None else message.get('content', '')

    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (auto-detect if not specified)"
    )
    parser.add_argument(
        "--audio", type=str, default="assets/mmau_test.wav", help="Path to audio file"
    )
    return parser.parse_args()


def main(args):
    # Build question
    question = "Which of the following best describes the male vocal in the audio?"
    choices = ["Soft and melodic", "Aggressive and talking", "High-pitched and singing", "Whispering"]
    question_text = f"{question}\nPlease choose the answer from the following options, do not provide any additional explanations or content:\n"
    question_text = f"{question}\nPlease choose the answer from the following options, output the thinking process in <think> </think> and final answer in <answer> </answer>:\n"
    question_text = f"{question}\nPlease choose the answer from the following options, output the thinking process in <think> </think> and final answer in <answer> </answer> (Tips: the ground truth is Aggressive and talking):\n"
    for i, choice in enumerate(choices):
        question_text += f"{chr(65+i)}. {choice}\n"

    print(f"Loading audio: {args.audio}")
    print("Sending request...")

    # Call API
    result = call_audio_llm_api(
        audio_path=args.audio,
        question_text=question_text,
        model_name=args.model,
    )

    # Extract text
    text = extract_text_from_result(result)

    print("\n\nFull response:", text)


if __name__ == "__main__":
    args = parse_args()
    main(args)
