# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import base64
import json
import os

import requests

# vLLM API server configuration
api_base = "http://localhost:8000/v1"
api_base = "http://l20-2:9999/v1"
# api_base = "http://localhost:9999/v1"

def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (auto-detect if not specified)"
    )
    parser.add_argument(
        "--audio", type=str, default="assets/mmau_test.wav", help="Path to audio file"
    )
    return parser.parse_args()


def get_model_name(api_base):
    """Get the first available model from the server."""
    response = requests.get(f"{api_base}/models")
    response.raise_for_status()
    models = response.json()
    return models["data"][0]["id"]


def load_audio_base64(audio_path):
    """Load audio file and encode to base64."""
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode("utf-8")


def main(args):
    # Get model name
    model = args.model or get_model_name(api_base)
    print(f"Using model: {model}")

    # Load audio file
    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    print(f"Loading audio: {audio_path}")
    audio_base64 = load_audio_base64(audio_path)

    # Build question
    question = "Which of the following best describes the male vocal in the audio?"
    choices = ["Soft and melodic", "Aggressive and talking", "High-pitched and singing", "Whispering"]
    question_text = f"{question}\nPlease choose the answer from the following options, do not provide any additional explanations or content:\n"
    # question_text = f"{question}\nPlease choose the answer from the following options, output the thinking process in <think> </think> and final answer in <answer> </answer>:\n"
    for i, choice in enumerate(choices):
        question_text += f"{chr(65+i)}. {choice}\n"

    # Chat Completion API with audio (offline)
    # Align with mmau_test_offline() in examples-vllm_r1.py
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
            {"role": "assistant", "content": "<think>\n"},
        ],
        "stream": False,
        "max_tokens": 16000,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "stop_token_ids": [151665],
        "continue_final_message": True,
        "add_generation_prompt": False,
        "skip_special_tokens": False,
    }

    headers = {"Content-Type": "application/json"}
    print("\nSending request...")
    response = requests.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()

    result = response.json()

    choice_data = result['choices'][0]
    message = choice_data['message']

    # Extract text: try tts_content.tts_text first, fallback to content
    # (aligned with stepaudior1vllm.py offline() method)
    text = message.get('tts_content', {}).get('tts_text', None)
    text = text if text is not None else message.get('content', '')

    logprobs = choice_data.get('logprobs', None)
    prompt_logprobs = choice_data.get('prompt_logprobs', None)

    print("\n\nFull response:", text)


if __name__ == "__main__":
    args = parse_args()
    main(args)