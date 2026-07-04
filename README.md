# raranker-api
Reranker API

## Prepare

```
conda create -n rerank-api python=3.11 -y
conda activate rerank-api
```

```bash
pip install -r requirements.txt
```

## Run

```
uvicorn main:app --reload --host 0.0.0.0 --port 8886
or on windows
./main.bat
or on linux
./main.sh
```

## Use

```
http://localhost:8886/v1/rerank
```

## Supported Models

- bce-reranker-base_v1
