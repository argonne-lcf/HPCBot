import time
from typing import Optional, Literal, Union
import statistics
import openai
from openai import OpenAI
import requests
import argparse
import logging
import platform
import psutil
import json
from pathlib import Path
import uuid
from datetime import datetime
import subprocess
import shutil

ModelType = Literal["openai", "ollama"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_nvidia_gpu_info():
    """Get NVIDIA GPU information using nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line:
                name, memory, driver = line.split(", ")
                gpus.append(
                    {
                        "name": name,
                        "memory": float(memory) / 1024,  # Convert MB to GB
                        "driver": driver,
                    }
                )
        return gpus
    except (subprocess.SubprocessError, ValueError):
        return []


def get_system_info():
    """Get system information including CPU, GPU, and memory."""
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": {
            "processor": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total": psutil.virtual_memory().total / (1024**3),  # GB
            "available": psutil.virtual_memory().available / (1024**3),  # GB
        },
        "gpu": get_nvidia_gpu_info(),
    }
    return info


def save_benchmark_results(results: dict, system_info: dict, output_dir: Path):
    """Save benchmark results and system info to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    filename = f"benchmark_{timestamp}_{run_id}.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    full_results = {
        "timestamp": timestamp,
        "run_id": run_id,
        "system_info": system_info,
        "benchmark_results": results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    return output_file


def save_model_response(response_text: str, model_info: dict, output_dir: Path):
    """Save model response to a log file with timestamp and model info."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "responses.log"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"\n{'='*80}\n"
        f"Timestamp: {timestamp}\n"
        f"Model: {model_info['model_name']} ({model_info['model_type']})\n"
        f"Prompt: {model_info['prompt']}\n"
        f"{'-'*80}\n"
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(header)
        f.write(response_text)
        f.write("\n")


def estimate_tokens(text: str) -> float:
    """Estimate number of tokens from character count using 4 chars/token ratio."""
    return len(text) / 4


def benchmark_generation(
    model_name: str,
    prompt: str,
    model_type: ModelType,
    num_runs: int = 3,
    max_tokens: int = 100,
    api_base: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Benchmark an LLM's generation performance.

    Args:
        model_name: Name of the model to benchmark (e.g. "gpt-3.5-turbo" or "llama2")
        prompt: Input prompt to generate from
        model_type: Type of model - either "openai" or "ollama"
        num_runs: Number of benchmark runs to average over
        max_tokens: Maximum number of tokens to generate
        api_base: Optional API base URL (mainly for Ollama)
        output_dir: Optional directory to save response logs

    Returns:
        Dictionary containing benchmark results
    """
    times = []
    token_counts = []

    # Set up client
    if model_type == "openai":
        client = OpenAI(base_url=api_base) if api_base else OpenAI()
        effective_api_base = str(client.base_url)  # Convert URL to string
    elif model_type == "ollama":
        effective_api_base = api_base or "http://localhost:11434"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_info = {
        "model_name": model_name,
        "model_type": model_type,
        "prompt": prompt,
    }

    logger.info(f"Running {num_runs} benchmark iterations...")
    for i in range(num_runs):
        logger.info(f"Run {i+1}/{num_runs}...")

        start_time = time.time()

        if model_type == "openai":
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            generated_text = response.choices[0].message.content
            # Use character-based estimation instead of API's token count
            generated_tokens = estimate_tokens(generated_text)

        else:  # ollama
            response = requests.post(
                f"{effective_api_base}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            generated_text = data["response"]
            generated_tokens = estimate_tokens(generated_text)

        # Save response if output directory is provided
        if output_dir:
            save_model_response(generated_text, model_info, output_dir)

        end_time = time.time()
        duration = end_time - start_time
        tokens_per_second = generated_tokens / duration

        times.append(duration)
        token_counts.append(generated_tokens)

        logger.info(
            f"Generated ~{int(generated_tokens)} tokens in {duration:.2f}s "
            f"({tokens_per_second:.2f} tokens/s)"
        )

    # Calculate statistics
    avg_tokens = statistics.mean(token_counts)
    avg_time = statistics.mean(times)
    avg_tokens_per_second = avg_tokens / avg_time

    # Calculate tokens per second for each run
    tokens_per_second_runs = [
        tokens / time for tokens, time in zip(token_counts, times)
    ]

    # Find max and min performance
    max_tokens_per_second = max(tokens_per_second_runs)
    min_tokens_per_second = min(tokens_per_second_runs)

    results = {
        "model_name": model_name,
        "model_type": model_type,
        "api_base": effective_api_base,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "num_runs": num_runs,
        "average_tokens": avg_tokens,
        "average_time": avg_time,
        "tokens_per_second": avg_tokens_per_second,
        "max_tokens_per_second": max_tokens_per_second,
        "min_tokens_per_second": min_tokens_per_second,
        "individual_runs": [
            {
                "tokens": t,
                "time": tim,
                "tokens_per_second": t / tim,
            }
            for t, tim in zip(token_counts, times)
        ],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM generation performance")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Name of the model (e.g., gpt-3.5-turbo for OpenAI or llama2 for Ollama)",
        dest="model_name",
    )
    parser.add_argument(
        "--server",
        "-s",
        type=str,
        choices=["openai", "ollama"],
        required=True,
        help="Server to use (openai or ollama)",
        dest="model_type",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Write a motivational poem about using AI for science.",
        help="Prompt to use for generation",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=3,
        help="Number of benchmark runs to perform",
        dest="num_runs",
    )
    parser.add_argument(
        "--tokens",
        "-t",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
        dest="max_tokens",
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        help="API base URL (for OpenAI or Ollama). Defaults to standard OpenAI URL or http://localhost:11434 for Ollama",
        dest="api_base",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Directory to save benchmark results (default: results)",
        dest="output_dir",
    )

    args = parser.parse_args()

    # Get system information
    system_info = get_system_info()

    # Create output directory
    output_dir = Path(args.output_dir)

    # Run benchmark
    results = benchmark_generation(
        model_name=args.model_name,
        prompt=args.prompt,
        model_type=args.model_type,
        num_runs=args.num_runs,
        max_tokens=args.max_tokens,
        api_base=args.api_base,
        output_dir=output_dir,  # Pass output directory to save responses
    )

    # Save results to file
    output_file = save_benchmark_results(results, system_info, output_dir)

    # Log results to console
    logger.info("\nBenchmark Results:")
    logger.info(f"Model: {results['model_name']} ({results['model_type']})")
    logger.info(f"API Base: {results['api_base']}")
    logger.info(f"Max tokens: {results['max_tokens']}")
    logger.info(f"Prompt: {results['prompt']}")
    logger.info(f"Average tokens generated: {results['average_tokens']:.1f}")
    logger.info(f"Average time: {results['average_time']:.2f}s")
    logger.info(f"Average tokens per second: {results['tokens_per_second']:.2f}")
    logger.info("\nPerformance Range:")
    logger.info(f"Max tokens per second: {results['max_tokens_per_second']:.2f}")
    logger.info(f"Min tokens per second: {results['min_tokens_per_second']:.2f}")
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
