from FlagEmbedding import LayerWiseFlagLLMReranker, FlagLLMReranker, FlagReranker
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
bge_reranker_v2_minicpm_layerwise = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True)
bge_reranker_v2_gemma = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)
bge_reranker_v2_m3 = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)


class DocumentRankerManager(BaseModel):
    query_text: str
    unranked_docs: list
    ranker_model: str


@app.post("/rerank/")
def rerank_docs(manager: DocumentRankerManager) -> list:
    query_text = manager.query_text
    ranker_model = manager.ranker_model
    unranked_docs = manager.unranked_docs
    if ranker_model == 'BAAI/bge-reranker-v2-minicpm-layerwise':
        cross = [(query_text, doc) for doc in unranked_docs]
        ce_scores = bge_reranker_v2_minicpm_layerwise.compute_score(cross, batch_size=1, cutoff_layers=[28])
        return ce_scores
    elif ranker_model == 'BAAI/bge-reranker-v2-gemma':
        cross = [(query_text, doc) for doc in unranked_docs]
        ce_scores = bge_reranker_v2_gemma.compute_score(cross, batch_size=1)
        return ce_scores
    elif ranker_model == 'BAAI/bge-reranker-v2-m3':
        cross = [(query_text, doc) for doc in unranked_docs]
        ce_scores = bge_reranker_v2_m3.compute_score(cross, batch_size=1)
        return ce_scores
    else:
        cross = [(query_text, doc) for doc in unranked_docs]
        ce_scores = bge_reranker_v2_m3.compute_score(cross, batch_size=1, cutoff_layers=[28])
        return ce_scores
