call conda activate rerank-api
uvicorn main:app --reload --host 0.0.0.0 --port 7987