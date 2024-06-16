from BCEmbedding import RerankerModel
from FlagEmbedding import LayerWiseFlagLLMReranker, FlagLLMReranker, FlagReranker
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
# bge_reranker_v2_minicpm_layerwise = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True)
# bge_reranker_v2_gemma = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)
# bge_reranker_v2_m3 = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
bce_reranker_base_v1 = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")


class DocumentRankerManager(BaseModel):
    query_text: str
    unranked_docs: list
    ranker_model: str


@app.post("/v1/rerank")
def rerank_docs(manager: DocumentRankerManager) -> list:
    """
    Reranks a list of documents based on a given query text and ranker model.

    Args:
        manager (DocumentRankerManager): An object containing the query text, unranked documents, and ranker model.

    Returns:
        list: A list of reranked documents based on the given query text and ranker model.
    """
    query_text = manager.query_text
    ranker_model = manager.ranker_model
    unranked_docs = manager.unranked_docs
    if ranker_model == 'BAAI/bge-reranker-v2-minicpm-layerwise':
        bge_reranker_v2_minicpm_layerwise = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise',
                                                                     use_fp16=True)
        cross = [(query_text, doc) for doc in unranked_docs]
        ce_scores = bge_reranker_v2_minicpm_layerwise.compute_score(cross, batch_size=1, cutoff_layers=[28])
        return ce_scores
    # elif ranker_model == 'BAAI/bge-reranker-v2-gemma':
    #     cross = [(query_text, doc) for doc in unranked_docs]
    #     ce_scores = bge_reranker_v2_gemma.compute_score(cross, batch_size=1)
    #     return ce_scores
    # elif ranker_model == 'BAAI/bge-reranker-v2-m3':
    #     cross = [(query_text, doc) for doc in unranked_docs]
    #     ce_scores = bge_reranker_v2_m3.compute_score(cross, batch_size=1)
    #     return ce_scores
    elif ranker_model == 'maidalun1020/bce-reranker-base_v1':
        cross = [(query_text, doc) for doc in unranked_docs]
        ce_scores = bce_reranker_base_v1.compute_score(cross, batch_size=1)
        return ce_scores
    else:
        bge_reranker_v2_m3 = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        cross = [(query_text, doc) for doc in unranked_docs]
        ce_scores = bge_reranker_v2_m3.compute_score(cross, batch_size=1, cutoff_layers=[28])
        return ce_scores
