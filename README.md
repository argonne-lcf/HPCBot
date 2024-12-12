# RAG/ Finetune

## QA generators:
The QA_generator_*.ipynb [notebooks](notebooks/QA_generator_Polaris.py) generate artificial question/answer JSON files based on a given golden context. These include:
- id: original-file-location_seed_task_x_y, where x is the id of the golden context chunk, and y is the id of the question generated (if we generate 3 questions, then y = 0, 1, 2).
- context: including distractor contexts and might include the golden context (probability p = 0.8).
- golden_context: the context that was used to generate QA.
- cot_answer: includes the full chain of thought answer.
- answer: only includes the final answer.
  
[Example file](output/QA_polaris_md_llama3-chatqa:70b_1500_3.json)

## Pre-finetuning processing
- The [pre-FT-processing.ipynb notebook](notebooks/pre-FT-processing.ipynb) is used to generate a data file for finetuning. Here we use autotrain for finetuning, so the output file has to have a "text" column. Each row will be in the format of ###Human: question ###Assistant: answer. [Example](data/train/QA_only/train_with_QA_only.jsonl)
- Use autotrain for finetuning: autotrain --config "config file location". [Config file example](configs/llama3-1-8b-sft-500-2-with-QA-only.yml)

## Post-finetuning processing
- After finetuning, we will have adapter files. To merge the adapters with a base LLM model, we use [post-FT-processing.ipynb](notebooks/post-FT-processing.ipynb).
- To use the new finetuned model with Ollama:
  - Create a modelfile. [How to](https://www.gpu-mart.com/blog/custom-llm-models-with-ollama-modelfile). [Example](ollama_modelfiles/HPCBot-LLama3-1-8B-with-QA-only.modelfile)
  - Use [llama.cpp](https://github.com/ggerganov/llama.cpp) to convert the trained model to a GGUF file.
  - Create a new Ollama model with the new modelfile and GGUF file.

## Evaluate with autorag
Create an autorag corpus and QA parquet using the autorag notebook. Autorag can compare multiple LLM models, prompts, retrieval methods, top_k, etc.

## Manually test LLM models' answers
Using the [evaluate.ipynb](notebooks/evaluate.ipynb) notebook, we can test different models with a set of fixed questions. [Example](QA_eval_llama-3-1.json)

## Run local RAG model
To run a local RAG model, use the [local_RAG_md.ipynb notebook](notebooks/local_RAG_md.ipynb).