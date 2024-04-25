# raranker-api
Reranker API

## Prepare

```
conda create -n raranker-api python=3.11 -y
conda activate raranker-api
```

```bash
pip install -r requirements.txt
``

## Run

```
uvicorn main:app --reload --host 0.0.0.0 --port 8765
or on windows
./main.bat
or on linux
./main.sh
```
## Test

```
python client.py
```