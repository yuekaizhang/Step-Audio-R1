# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test MMAU benchmark using remote vLLM API server."""

import argparse
import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
import torchaudio
from tqdm import tqdm

# Default API configuration
DEFAULT_API_BASE = "http://l20-1:19000/v1"


def get_model_name(api_base: str) -> str:
    """Get the first available model from the server."""
    response = requests.get(f"{api_base}/models")
    response.raise_for_status()
    models = response.json()
    return models["data"][0]["id"]


def load_audio_base64(audio_path: str, max_audio_in_seconds: Optional[int] = None) -> str:
    """Load audio file and encode to base64.

    Args:
        audio_path: Path to the audio file
        max_audio_in_seconds: Maximum audio duration in seconds. If None, no truncation.

    Returns:
        Base64 encoded audio string
    """
    if max_audio_in_seconds is None:
        # No truncation, just read and encode
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode("utf-8")

    # Load audio with torchaudio for truncation
    waveform, sample_rate = torchaudio.load(audio_path)

    # Calculate max samples and truncate
    max_samples = max_audio_in_seconds * sample_rate
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # Save truncated audio to bytes buffer
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def call_audio_llm_api(
    audio_path: str,
    question_text: str,
    answer_text: str = "<think>\n",
    api_base: str = DEFAULT_API_BASE,
    model_name: Optional[str] = None,
    max_tokens: int = 16000,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    stop_token_ids: Optional[list] = None,
    max_audio_in_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Call the audio LLM API with the given audio and question."""
    if stop_token_ids is None:
        stop_token_ids = [151665]

    # Auto-detect model if not specified
    model = model_name or get_model_name(api_base)

    # Validate audio file
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load and encode audio
    audio_base64 = load_audio_base64(audio_path, max_audio_in_seconds)

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
    """Extract text from API result."""
    choice_data = result['choices'][0]
    message = choice_data['message']

    # Try tts_content.tts_text first, fallback to content
    text = message.get('tts_content', {}).get('tts_text', None)
    text = text if text is not None else message.get('content', '')

    return text


def extract_answer(output_str: str) -> str:
    """Extract answer from model output.

    If there's <think>...</think>, return the part after </think>.
    Otherwise, return the full response.
    """
    # Check if there's a </think> tag
    think_end_match = re.search(r"</think>\s*", output_str, re.DOTALL)
    if think_end_match:
        # Return everything after </think>
        return output_str[think_end_match.end():].strip()

    # No think tag, return the full response
    return output_str.strip()


def extract_think(output_str: str) -> str:
    """Extract content before </think> tag in the model output.

    Since <think> is in the input prompt, the model output starts with thinking content directly.
    """
    match = re.search(r"(.*?)</think>", output_str, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_question(obj_dict: Dict[str, Any], template: str = "qa") -> str:
    """Format question text from MMAU data item."""
    question = obj_dict['question']
    choices = obj_dict['choices']
    if template == "qa":
        question_text = f"{question}\nPlease choose the answer from the following options, do not provide any additional explanations or content:\n"
        for i, choice in enumerate(choices):
            question_text += f"{choice}\n"
    elif template == "caption":
        question_text = "Listen to the provided audio and produce a very detailed audio caption."
    elif template == "open_qa":
        question_text = question
    else:
        raise ValueError(f"Invalid template: {template}")
    return question_text


def parse_args():
    parser = argparse.ArgumentParser(description="Test MMAU with remote vLLM API")
    parser.add_argument("--api_base", type=str, default=DEFAULT_API_BASE, help="API server base URL")
    parser.add_argument("--model", type=str, default=None, help="Model name (auto-detect if not specified)")
    parser.add_argument("--data_file", type=str, required=True, help="MMAU test data file (JSON)")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--out_file", type=str, required=True, help="Output file for results (JSON)")
    parser.add_argument("--max_tokens", type=int, default=16000, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if output exists")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (exclusive)")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking")
    parser.add_argument("--template", type=str, default="qa", choices=["qa", "caption", "open_qa"], help="Template for question formatting")
    parser.add_argument("--max_audio_in_seconds", type=int, default=29, help="Max audio duration in seconds (truncate if longer)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if needed
    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load existing results for resume functionality
    existing_results = []
    processed_ids = set()
    if not args.force and os.path.exists(args.out_file) and os.path.getsize(args.out_file) > 0:
        try:
            with open(args.out_file, "r") as f:
                existing_results = json.load(f)
            processed_ids = {item.get("audio_id") for item in existing_results if item.get("audio_id")}
            print(f"Resuming: loaded {len(existing_results)} existing results, {len(processed_ids)} unique audio_ids")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load existing results ({e}), starting fresh")
            existing_results = []
            processed_ids = set()

    # Load test data
    with open(args.data_file, "r") as f:
        datas = json.load(f)

    # Apply index range if specified
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(datas)
    datas = datas[start_idx:end_idx]

    # Filter out already processed samples
    if processed_ids:
        original_count = len(datas)
        datas = [item for item in datas if item.get("audio_id") not in processed_ids]
        print(f"Skipping {original_count - len(datas)} already processed samples")

    print(f"Processing {len(datas)} samples (index {start_idx} to {end_idx})")

    if len(datas) == 0:
        print("All samples already processed. Use --force to regenerate.")
        return

    # Get model name once
    model_name = args.model or get_model_name(args.api_base)
    print(f"Using model: {model_name}")

    final_output = existing_results.copy()

    for item in tqdm(datas, desc="Processing"):
        audio_path = os.path.join(args.audio_dir, item["audio_id"])
        question_text = format_question(item, template=args.template)
        try:
            result = call_audio_llm_api(
                audio_path=audio_path,
                question_text=question_text,
                answer_text="<think>\n" if args.enable_thinking else "",
                api_base=args.api_base,
                model_name=model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_audio_in_seconds=args.max_audio_in_seconds,
            )

            # Extract full response
            full_response = extract_text_from_result(result)

            # Extract answer and thinking
            model_answer = extract_answer(full_response)
            model_think = extract_think(full_response)
            print(f"model_answer: {model_answer}")
            print(f"model_response: {full_response}")
            # Build result record
            result_record = item.copy()
            result_record["model_output"] = model_answer
            result_record["model_think"] = model_think
            result_record["model_response"] = full_response

        except Exception as e:
            print(f"\nError processing {item.get('audio_id', 'unknown')}: {e}")
            result_record = item.copy()
            result_record["model_output"] = ""
            result_record["model_think"] = ""
            result_record["model_response"] = f"ERROR: {str(e)}"

        final_output.append(result_record)

        # Save intermediate results periodically
        if len(final_output) % 10 == 0:
            with open(args.out_file, "w") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)

    # Save final results
    with open(args.out_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.out_file}")
    print(f"Total processed: {len(final_output)}")


if __name__ == "__main__":
    main()
