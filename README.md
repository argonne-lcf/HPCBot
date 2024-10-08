# RAG/ Finetune

## QA generators:
The QA_generator_*.ipynb notebooks are to used to generate artificial questions/answers json file based on given golden context. Which include:
- id: original-file-location_seed_task_x_y. Where x is the id of the golden context chunk, y is the id of the question generated (if we generate 3 questions then y = 0, 1, 2)
- context: inculding distractor contexts and might include golden context (probility p = 0.8)
- golden_context: the context that was used to generate QA
- cot_answer: include full chain of thought answer
- answer: only include the final answer
  
[Example file](output/QA_polaris_md_llama3-chatqa:70b_1500_3.json)

## Pre fintuning processing
- The pre-FT-processing.ipynb notebook is used to generate data file for finetuning. Here we use autotrain for finetuning, so the output file has to have a "text" column. Each rows will be in the format of ###Human: question ###Assistant: answer. Example: data/train/QA_only/train_with_QA_only.jsonl
- Use autotrain for finetuning: autotrain --config "config file location". Config file example: configs/llama3-1-8b-sft-500-2-with-QA-only.yaml
## Post fintuning processing
- After finetuning we will have adapters file, to merge the adapters to a base LLM model, we use post-FT-processing.ipynb notebook.
- To use new finetuned model with Ollama:
  - create a modelfile. [How to](https://www.gpu-mart.com/blog/custom-llm-models-with-ollama-modelfile). Example:
  - use [llama.cpp](https://github.com/ggerganov/llama.cpp) to convert trained model to GGUF file
  - create new Ollama model with new modelfile and GGUF file.
##
