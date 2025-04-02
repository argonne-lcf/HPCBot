# hpcbot/cli.py
import argparse
import yaml
import os
import logging


def setup_logging(level_name="INFO"):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,  # This ensures we override any existing configuration
    )


# Set default logging level
setup_logging()

from hpcbot.chat import ChatBot
from hpcbot.generate import QAContextDistractors, QAAnswerDistractors

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file=None):
        self.config = {}
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file):
        """Load configuration from a YAML file."""
        try:
            with open(config_file, "r") as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def get_config(self, key, default=None):
        """Get a configuration value by key, with an optional default."""
        return self.config.get(key, default)


def validate_args(args, config):
    """Validate command-line arguments and configuration."""
    if args.command == "assist" and not (
        args.document_dir or config.get_config("document_dir")
    ):
        raise ValueError(
            "Document directory must be provided via --document_dir or --config."
        )
    if args.command == "generate" and not (
        args.document_dir or config.get_config("document_dir")
    ):
        raise ValueError(
            "Document directory must be provided via --document_dir or --config."
        )
    if args.command == "generate" and not (
        args.out_dir or config.get_config("out_dir")
    ):
        raise ValueError("Output directory must be provided via --out_dir or --config.")


def main():
    parser = argparse.ArgumentParser(
        description="HPCBot: A CLI tool for document-based QA and generation."
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Assist command
    assist_parser = subparsers.add_parser(
        "assist", help="Chat with the document-based bot."
    )
    assist_parser.add_argument("question", type=str, help="The question to ask.")
    assist_parser.add_argument(
        "--document_dir", type=str, help="Directory containing documents."
    )
    assist_parser.add_argument(
        "--config", type=str, help="Path to the configuration file."
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate QA pairs from documents."
    )
    generate_parser.add_argument(
        "--document_dir", type=str, help="Directory containing documents."
    )
    generate_parser.add_argument(
        "--out_dir", type=str, help="Directory to save QA pairs."
    )
    generate_parser.add_argument(
        "--config", type=str, help="Path to the configuration file."
    )

    args = parser.parse_args()

    # Update logging configuration with command line argument
    setup_logging(args.log_level)
    logger.info("Logging level set to: %s", args.log_level)

    # Load configuration if provided
    config = Config(args.config) if args.config else Config()
    logger.info("Logging level: %s", args.log_level)
    # Set up configuration values
    is_local = config.get_config("is_local")
    access_token = config.get_config("access_token")
    base_url = config.get_config("base_url")
    model = config.get_config("model", "llama3.1")  # Default model
    document_dir = args.document_dir or config.get_config("document_dir")
    distractor = config.get_config(
        "distractor", "answer"
    )  # Default to answer-based distractors

    # Handle local Ollama configuration
    if is_local:
        access_token = "ollama"
        base_url = "http://localhost:11434/v1"
    elif not access_token:
        # Load access token from file if not provided
        token_file = (
            "/lus/eagle/projects/HPCBot/workspace/hpcbot/hpcbot/access_token.txt"
        )
        if os.path.exists(token_file):
            with open(token_file, "r") as file:
                access_token = file.read().strip()
        else:
            raise ValueError("Access token not provided and token file not found.")

    # Validate arguments
    validate_args(args, config)

    # Execute the command
    if args.command == "assist":
        bot = ChatBot(model, document_dir, access_token, base_url)
        answer = bot.ask(args.question)
        logger.info(answer)
    elif args.command == "generate":
        out_dir = args.out_dir or config.get_config("out_dir")
        if distractor == "answer":
            generator = QAAnswerDistractors(
                model, access_token, base_url, document_dir, out_dir
            )
        else:
            generator = QAContextDistractors(
                model, access_token, base_url, document_dir, out_dir
            )
        generator.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
