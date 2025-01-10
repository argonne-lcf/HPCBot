import sys
import os
import argparse
sys.path.append("..")
import unittest
from workspace.package.hpcbot.generate import QAContextDistractors, QAAnswerDistractors

# Custom argument parser
parser = argparse.ArgumentParser(description='Give --path --filetype --output ...')
parser.add_argument('--path', type=str, default='../data/', help='data path')
parser.add_argument('--filetype', type=str, default='md', help='data input filetype')
parser.add_argument('--output', type=str, default='../output/tests/', help='output path')
parser.add_argument('--num', type=int, default=3, help='num of questions')
parser.add_argument('--num_distractors', type=int, default=3, help='num of distractors for context')
parser.add_argument('--num_answer', type=int, default=4, help='num of wrong answers')

# Parse arguments early
args, remaining_argv = parser.parse_known_args()

# Ensure unittest ignores custom arguments
sys.argv = [sys.argv[0]] + remaining_argv

class TestQAContextDistractors(unittest.TestCase):
    def setUp(self):
        self.processor = QAContextDistractors()

    def test_generate_questions(self):
        datasets = self.processor.run(path = args.path, file_type = args.filetype, output= os.path.join(args.output, "test_qa_context.json"), stop_early=True)
        self.assertGreater(len(datasets), 0)

class TestQAAnswerDistractors(unittest.TestCase):
    def setUp(self):
        self.processor = QAAnswerDistractors()

    def test_generate_questions(self):
        datasets = self.processor.run(path = args.path, file_type = args.filetype, output= os.path.join(args.output, "test_qa_answer.json"),stop_early=True)
        self.assertGreater(len(datasets), 0)

if __name__ == "__main__":
    unittest.main()