#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate rerank-api
PYTORCH_NVML_BASED_CUDA_CHECK=-1 CUDA_VISIBLE_DEVICES=2 uvicorn openai_api:app --reload --host 0.0.0.0 --port 8886
