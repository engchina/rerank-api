import os
from typing import Union

from BCEmbedding import RerankerModel
from fastapi import FastAPI, Request
from pydantic import BaseModel

# 设置环境变量，指定模型和数据集路径
MODEL_PATH = "/app/models/"
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"

# 优化点 2：减少常量加载时间
WORKER_API_EMBEDDING_BATCH_SIZE = int(os.getenv("FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE", 4))

app = FastAPI()
# bce_reranker_base_v1 = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")
bce_reranker_base_v1 = RerankerModel(model_name_or_path=MODEL_PATH + "maidalun1020/bce-reranker-base_v1")


class JinaRankerBody(BaseModel):
    model: str
    query: str
    top_n: int = 3
    documents: list
    return_documents: bool = True


@app.post("/v1/rerank")
@app.post("/v2/rerank")
async def rerank_docs(request: Request, body: JinaRankerBody):
    """
    Reranks a list of documents based on a given query text and ranker model.
    Supports both original and Jina AI API formats.
    """
    # 打印原始request body
    raw_body = await request.body()
    print(f"Raw request body: {raw_body.decode('utf-8')}")

    # 打印request中的所有key-value
    print("Request headers:")
    for key, value in request.headers.items():
        print(f"{key}: {value}")

    # 获取请求头中的credentials
    credentials = request.headers.get("Authorization")
    print(f"Credentials: {credentials}")
    query = body.query
    documents = body.documents

    cross = [(query, doc) for doc in documents]
    ce_scores = bce_reranker_base_v1.compute_score(cross, batch_size=1)

    print(f"{ce_scores=}")

    # 为Jina AI API格式构建返回结果
    results = []
    for idx, score in enumerate(ce_scores):
        results.append({
            "credentials": "set to no dict to skip update",
            "index": int(idx),
            "relevance_score": float(score),
            "document": documents[idx]
        })
    print(f"{results=}")
    return {"results": results}
