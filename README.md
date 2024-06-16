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
uvicorn main:app --reload --host 0.0.0.0 --port 7987
or on windows
./main.bat
or on linux
./main.sh
```

## Use

```
http://localhost:7987/v1/rerank
```

## Supported Models

- bce-reranker-base_v1
