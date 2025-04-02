import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
from hpcbot.generate import QAContextDistractors, QAAnswerDistractors


class TestQAContextDistractors(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_doc_dir = os.path.join(self.temp_dir, "test_docs")
        self.test_output_dir = os.path.join(self.temp_dir, "test_output")
        os.makedirs(self.test_doc_dir)
        os.makedirs(self.test_output_dir)

        # Create multiple test markdown files to ensure enough chunks
        for i in range(3):  # Create 3 test files
            test_file = os.path.join(self.test_doc_dir, f"test_{i}.md")
            with open(test_file, "w") as f:
                f.write(f"This is test document {i} for HPCBot testing.")

        # Initialize the processor with test values
        self.processor = QAContextDistractors(
            model="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            document_dir=self.test_doc_dir,
            out_dir=os.path.join(self.test_output_dir, "test_qa_context.json"),
        )

    def tearDown(self):
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("openai.OpenAI")
    def test_generate_questions(self, mock_openai):
        # Mock the OpenAI client responses
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock the chat completion response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="1. What is HPCBot?\n2. How does it work?\n3. What are its features?"
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the necessary methods
        self.processor.client = mock_client

        # Test the run method with minimal parameters
        datasets = self.processor.run(
            num_questions=1, num_distractors=1, stop_early=True
        )

        self.assertIsInstance(datasets, list)
        self.assertGreater(len(datasets), 0)

        # Verify the mock was called
        mock_client.chat.completions.create.assert_called()


class TestQAAnswerDistractors(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_doc_dir = os.path.join(self.temp_dir, "test_docs")
        self.test_output_dir = os.path.join(self.temp_dir, "test_output")
        os.makedirs(self.test_doc_dir)
        os.makedirs(self.test_output_dir)

        # Create multiple test markdown files to ensure enough chunks
        for i in range(3):  # Create 3 test files
            test_file = os.path.join(self.test_doc_dir, f"test_{i}.md")
            with open(test_file, "w") as f:
                f.write(f"This is test document {i} for HPCBot testing.")

        # Initialize the processor with test values
        self.processor = QAAnswerDistractors(
            model="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            document_dir=self.test_doc_dir,
            out_dir=os.path.join(self.test_output_dir, "test_qa_answer.json"),
        )

    def tearDown(self):
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("openai.OpenAI")
    def test_generate_questions(self, mock_openai):
        # Mock the OpenAI client responses
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock the chat completion response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="1. What is HPCBot?\n2. How does it work?\n3. What are its features?"
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the necessary methods
        self.processor.client = mock_client

        # Test the run method with minimal parameters
        datasets = self.processor.run(num_questions=1, num_answers=2, stop_early=True)

        self.assertIsInstance(datasets, list)
        self.assertGreater(len(datasets), 0)

        # Verify the mock was called
        mock_client.chat.completions.create.assert_called()


if __name__ == "__main__":
    unittest.main()
