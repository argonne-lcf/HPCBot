import time
from typing import Optional, Literal, Union
import statistics
import openai
from openai import OpenAI
import requests
import argparse
import logging

ModelType = Literal["openai", "ollama"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def benchmark_generation(
    model_name: str,
    prompt: str,
    model_type: ModelType,
    num_runs: int = 3,
    max_tokens: int = 100,
    api_base: Optional[str] = None,
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

    Returns:
        Dictionary containing benchmark results
    """
    times = []
    token_counts = []

    # Set up client
    if model_type == "openai":
        client = OpenAI()
    elif model_type == "ollama":
        api_base = api_base or "http://localhost:11434"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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
            generated_tokens = response.usage.completion_tokens

        else:  # ollama
            response = requests.post(
                f"{api_base}/api/generate",
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
            # Ollama doesn't provide token counts directly, so we estimate from response length
            # This is a rough approximation
            generated_tokens = len(data["response"].split()) * 1.3

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

    # Calculate best performance
    min_time = min(times)
    best_run_idx = times.index(min_time)
    best_tokens = token_counts[best_run_idx]
    best_tokens_per_second = best_tokens / min_time

    results = {
        "model_name": model_name,
        "model_type": model_type,
        "num_runs": num_runs,
        "average_tokens": avg_tokens,
        "average_time": avg_time,
        "tokens_per_second": avg_tokens_per_second,
        "best_time": min_time,
        "best_tokens": best_tokens,
        "best_tokens_per_second": best_tokens_per_second,
        "individual_runs": [
            {"tokens": t, "time": tim} for t, tim in zip(token_counts, times)
        ],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM generation performance")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model (e.g., gpt-3.5-turbo for OpenAI or llama2 for Ollama)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["openai", "ollama"],
        required=True,
        help="Type of model to benchmark",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short story about a robot:",
        help="Prompt to use for generation",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of benchmark runs to perform",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        help="API base URL (for Ollama, defaults to http://localhost:11434)",
    )

    args = parser.parse_args()

    results = benchmark_generation(
        model_name=args.model_name,
        prompt=args.prompt,
        model_type=args.model_type,
        num_runs=args.num_runs,
        max_tokens=args.max_tokens,
        api_base=args.api_base,
    )

    logger.info("\nBenchmark Results:")
    logger.info(f"Model: {results['model_name']} ({results['model_type']})")
    logger.info(f"Average tokens generated: {results['average_tokens']:.1f}")
    logger.info(f"Average time: {results['average_time']:.2f}s")
    logger.info(f"Average tokens per second: {results['tokens_per_second']:.2f}")
    logger.info("\nBest Performance:")
    logger.info(f"Best time: {results['best_time']:.2f}s")
    logger.info(f"Best tokens per second: {results['best_tokens_per_second']:.2f}")


if __name__ == "__main__":
    main()
