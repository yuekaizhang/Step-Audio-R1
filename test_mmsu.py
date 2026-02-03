# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test MMSU benchmark using remote vLLM API server."""

import argparse
import base64
import io
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Optional

import requests
import torchaudio
from datasets import load_dataset
from tqdm import tqdm

# Default API configuration
DEFAULT_API_BASE = "http://l20-2:9999/v1"


def get_model_name(api_base: str) -> str:
    """Get the first available model from the server."""
    response = requests.get(f"{api_base}/models")
    response.raise_for_status()
    models = response.json()
    return models["data"][0]["id"]


def load_audio_base64(audio_array, sample_rate: int, max_audio_in_seconds: Optional[int] = None) -> str:
    """Convert audio array to base64 encoded WAV.

    Args:
        audio_array: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        max_audio_in_seconds: Maximum audio duration in seconds. If None, no truncation.

    Returns:
        Base64 encoded audio string
    """
    import torch

    # Convert to tensor if needed
    if not isinstance(audio_array, torch.Tensor):
        waveform = torch.tensor(audio_array).unsqueeze(0)  # Add channel dimension
    else:
        waveform = audio_array.unsqueeze(0) if audio_array.dim() == 1 else audio_array

    # Truncate if needed
    if max_audio_in_seconds is not None:
        max_samples = max_audio_in_seconds * sample_rate
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

    # Save to bytes buffer
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def call_audio_llm_api(
    audio_base64: str,
    question_text: str,
    answer_text: str = "<think>\n",
    api_base: str = DEFAULT_API_BASE,
    model_name: Optional[str] = None,
    max_tokens: int = 16000,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    stop_token_ids: Optional[list] = None,
) -> Dict[str, Any]:
    """Call the audio LLM API with the given audio and question."""
    if stop_token_ids is None:
        stop_token_ids = [151665]

    # Auto-detect model if not specified
    model = model_name or get_model_name(api_base)

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

    If there's </think>, return the part after </think>.
    Otherwise, return the full response.
    """
    think_end_match = re.search(r"</think>\s*", output_str, re.DOTALL)
    if think_end_match:
        return output_str[think_end_match.end():].strip()
    return output_str.strip()


def extract_think(output_str: str) -> str:
    """Extract content before </think> tag in the model output."""
    match = re.search(r"(.*?)</think>", output_str, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_category(category_str: str) -> dict:
    """Parse category string like 'Perception-Linguistics-Phonology-Prosody' into components."""
    parts = category_str.split("-")
    return {
        "category": parts[0] if len(parts) > 0 else "",
        "sub_category": parts[1] if len(parts) > 1 else "",
        "sub_sub_category": parts[2] if len(parts) > 2 else "",
        "sub_discipline": parts[3] if len(parts) > 3 else "",
    }


def calculate_hierarchical_accuracy(results: list) -> dict:
    """Calculate accuracy at different hierarchy levels."""
    stats = {
        "category": defaultdict(lambda: {"correct": 0, "total": 0}),
        "sub_category": defaultdict(lambda: {"correct": 0, "total": 0}),
        "sub_sub_category": defaultdict(lambda: {"correct": 0, "total": 0}),
        "sub_discipline": defaultdict(lambda: {"correct": 0, "total": 0}),
        "category_sub": defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})),
    }

    for r in results:
        is_correct = r.get("is_correct", False)
        cat_parsed = r.get("category_parsed", {})
        cat = cat_parsed.get("category", "")
        sub = cat_parsed.get("sub_category", "")
        sub_sub = cat_parsed.get("sub_sub_category", "")
        discipline = cat_parsed.get("sub_discipline", "")

        # Update category level
        if cat:
            stats["category"][cat]["total"] += 1
            if is_correct:
                stats["category"][cat]["correct"] += 1

        # Update sub_category level
        if sub:
            stats["sub_category"][sub]["total"] += 1
            if is_correct:
                stats["sub_category"][sub]["correct"] += 1

        # Update sub_sub_category level
        if sub_sub:
            stats["sub_sub_category"][sub_sub]["total"] += 1
            if is_correct:
                stats["sub_sub_category"][sub_sub]["correct"] += 1

        # Update sub_discipline level
        if discipline:
            stats["sub_discipline"][discipline]["total"] += 1
            if is_correct:
                stats["sub_discipline"][discipline]["correct"] += 1

        # Update nested category -> sub_category
        if cat and sub:
            stats["category_sub"][cat][sub]["total"] += 1
            if is_correct:
                stats["category_sub"][cat][sub]["correct"] += 1

    # Calculate accuracies
    accuracy_stats = {}

    for level in ["category", "sub_category", "sub_sub_category", "sub_discipline"]:
        accuracy_stats[level] = {}
        for key, counts in stats[level].items():
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            accuracy_stats[level][key] = {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": acc,
            }

    # Nested category -> sub_category accuracy
    accuracy_stats["category_sub"] = {}
    for cat, subs in stats["category_sub"].items():
        accuracy_stats["category_sub"][cat] = {}
        for sub, counts in subs.items():
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            accuracy_stats["category_sub"][cat][sub] = {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": acc,
            }

    return accuracy_stats


def format_question(item: Dict[str, Any]) -> str:
    """Format question text from MMSU data item."""
    question = item.get('question', '')
    options = item.get('options', [])

    question_text = f"{question}\nPlease choose the answer from the following options, do not provide any additional explanations or content:\n"
    for option in options:
        question_text += f"{option}\n"

    return question_text


def parse_args():
    parser = argparse.ArgumentParser(description="Test MMSU with remote vLLM API")
    parser.add_argument("--api_base", type=str, default=DEFAULT_API_BASE, help="API server base URL")
    parser.add_argument("--model", type=str, default=None, help="Model name (auto-detect if not specified)")
    parser.add_argument("--hf_dataset_path", type=str, required=True, help="Path to HF MMSU dataset")
    parser.add_argument("--out_file", type=str, required=True, help="Output file for results (JSON)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--max_tokens", type=int, default=16000, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if output exists")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing (exclusive)")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking")
    parser.add_argument("--max_audio_in_seconds", type=int, default=29, help="Max audio duration in seconds")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if output file already exists
    if not args.force and os.path.exists(args.out_file) and os.path.getsize(args.out_file) > 0:
        print(f"Output file {args.out_file} already exists. Use --force to regenerate.")
        return

    # Create output directory if needed
    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load dataset
    print(f"Loading MMSU dataset from {args.hf_dataset_path}")
    dataset = load_dataset(args.hf_dataset_path, split='train')

    # Apply index range
    total_len = len(dataset)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else total_len

    print(f"Processing samples {start_idx} to {end_idx} (total: {end_idx - start_idx})")

    # Get model name once
    model_name = args.model or get_model_name(args.api_base)
    print(f"Using model: {model_name}")

    all_results = []

    for idx in tqdm(range(start_idx, end_idx), desc="Processing"):
        item = dataset[idx]

        # Extract fields from dataset item (MMSU format)
        # audio: dict with 'array' and 'sampling_rate'
        # key: identifier
        # question: the question text
        # category: category of the question
        # answer_index: index of correct answer in options (0-based)
        # options: list of answer options
        audio_data = item.get('audio', {})
        audio_array = audio_data.get('array', audio_data)
        sample_rate = audio_data.get('sampling_rate', 16000)
        question = item.get('question', '')
        options = item.get('options', [])
        answer_index = item.get('answer_index', 0)
        solution = options[answer_index] if options else ''
        category = item.get('category', '')
        key = item.get('key', str(idx))

        # Format question
        question_text = f"{question}\nPlease choose the answer from the following options, do not provide any additional explanations or content:\n"
        for option in options:
            question_text += f"{option}\n"

        try:
            # Convert audio to base64
            audio_base64 = load_audio_base64(audio_array, sample_rate, args.max_audio_in_seconds)

            # Call API
            result = call_audio_llm_api(
                audio_base64=audio_base64,
                question_text=question_text,
                answer_text="<think>\n" if args.enable_thinking else "",
                api_base=args.api_base,
                model_name=model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            # Extract response
            full_response = extract_text_from_result(result)
            model_answer = extract_answer(full_response)
            model_think = extract_think(full_response)

            print(f"model_answer: {model_answer}")

            # Parse category
            category_parsed = parse_category(category)

            # Check correctness
            is_correct = model_answer.strip() == solution.strip()

            result_record = {
                "key": key,
                "category": category,
                "category_parsed": category_parsed,
                "solution": solution,
                "model_output": model_answer,
                "model_think": model_think,
                "model_response": full_response,
                "is_correct": is_correct,
            }

            status = "✓" if is_correct else "✗"
            print(f"[{idx + 1}/{end_idx}] {status} answer={model_answer[:50]}...")

        except Exception as e:
            print(f"\nError processing index {idx}: {e}")
            result_record = {
                "key": key,
                "category": category,
                "category_parsed": parse_category(category),
                "solution": solution,
                "model_output": "",
                "model_think": "",
                "model_response": f"ERROR: {str(e)}",
                "is_correct": False,
            }

        all_results.append(result_record)

        # Save intermediate results periodically
        if len(all_results) % 10 == 0:
            _save_results(args.out_file, all_results, args)

    # Save final results
    _save_results(args.out_file, all_results, args)

    # Print summary
    total = len(all_results)
    correct = sum(1 for r in all_results if r["is_correct"])
    overall_accuracy = correct / total if total > 0 else 0.0

    print("=" * 80)
    print(f"Results saved to {args.out_file}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({correct}/{total})")

    # Print category-level accuracy
    hierarchical_stats = calculate_hierarchical_accuracy(all_results)
    print("-" * 80)
    print("Accuracy by Category:")
    for cat, stats in sorted(hierarchical_stats["category"].items()):
        print(f"  {cat}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    print("=" * 80)


def _save_results(out_file: str, results: list, args):
    """Save results to JSON file with metadata."""
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    overall_accuracy = correct / total if total > 0 else 0.0

    hierarchical_stats = calculate_hierarchical_accuracy(results)

    output_data = {
        "metadata": {
            "dataset_path": args.hf_dataset_path,
            "split": args.split,
            "total_samples": total,
            "correct": correct,
            "overall_accuracy": overall_accuracy,
        },
        "accuracy_by_category": hierarchical_stats["category"],
        "accuracy_by_sub_category": hierarchical_stats["sub_category"],
        "accuracy_by_sub_sub_category": hierarchical_stats["sub_sub_category"],
        "accuracy_by_sub_discipline": hierarchical_stats["sub_discipline"],
        "accuracy_by_category_sub": hierarchical_stats["category_sub"],
        "results": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
