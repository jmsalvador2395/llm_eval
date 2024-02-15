# LLM Evalutation On Semantic Overlap
This project is for the evaluation of various LLMs on the Semantic Overlap Summarization task

# Installation
- create a python environment >=3.10 and activate
- in the project directory run `pip install -e .`
- edit `cfg/config.yaml` to set LLMs to evaluate, metrics to compute, prompt files, datasets, etc.
- configure API keys
  - for OpenAI API key, you need to set it as an environment variable called `OPENAI_API_KEY`
  - for PaLM API, the key is provided in the config
- Execute code. run `llm_eval --help` for more details
