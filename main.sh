#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate rerank-api
uvicorn openai_api:app --reload --host 0.0.0.0 --port 7987