#!/bin/bash

# [1] create virtual env
yes | conda create -n tsc python=3.9 && conda activate tsc

# [2] install dependencies
pip install -r requirements.txt
yes | conda install tmux ruamel.yaml

# [3] make sure to modify your API keys in `.env` before running this step (example given in `.env.example`)
echo -e "\nNow exporting $(cat .env) as env variables.\n"
export $(cat .env)
echo "All set up! Now your api keys are:"
echo -e "\tOPENAI_API_KEY=\t\t[$(printenv OPENAI_API_KEY)]"
echo -e "\tHUGGINGFACE_API_KEY=\t[$(printenv HUGGINGFACE_API_KEY)]\n"