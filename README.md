# raranker-api
Reranker API

## Prepare

```
conda create -n raranker-api python=3.11 -y
conda activate raranker-api
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