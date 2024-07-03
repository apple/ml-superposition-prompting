import copy
from typing import Sequence, Tuple

import numpy as np
import torch
from src.prompting import SuperpositionPromptDef
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.prompting.superposition import *


def rank_by_attention(
    branch_idx_to_document_node_idxs, query_node_idxs, reduced_attentions
) -> Sequence[int]:
    assert reduced_attentions.ndim == 2

    n_branches = len(branch_idx_to_document_node_idxs)
    branch_weights = reduced_attentions.new_zeros(n_branches)
    for branch_idx, document_node_idxs in enumerate(branch_idx_to_document_node_idxs):
        didxs = torch.tensor(document_node_idxs)
        qidxs = torch.tensor(query_node_idxs)

        dd, qq = torch.meshgrid(didxs, qidxs, indexing="ij")
        branch_attentions = reduced_attentions[qq, dd].sum()
        branch_weights[branch_idx] = branch_attentions
    assert torch.all(torch.isfinite(branch_weights))
    _, indices = torch.sort(branch_weights, descending=False)
    return indices.tolist()


def attention_sort(
    tokenizer,
    model,
    prompt_string_obj: SuperpositionPromptDef,
    attention_sort_steps: int = 1,
    **kwargs,
) -> SuperpositionPromptDef:
    fii = prompt_string_obj.get_forkjoin_input_ids(tokenizer, model.device)
    contexts = list(fii.context_input_ids_list)

    prompt_string_obj = copy.deepcopy(prompt_string_obj)

    for i in range(attention_sort_steps):
        torch.cuda.empty_cache()
        preamble_len = fii.preamble_input_ids.numel()
        context_lens = [c.numel() for c in contexts]
        cumsum_context_lens = np.cumsum(context_lens).tolist()
        additives = [0] + cumsum_context_lens[:-1]
        branch_idx_to_document_node_idxs = [
            list(range(preamble_len + additives[i], preamble_len + additives[i] + c))
            for i, c in enumerate(context_lens)
        ]
        context_len_end_incl = branch_idx_to_document_node_idxs[-1][-1]
        query_len = fii.query_input_ids.numel()
        query_node_idxs = list(
            range(context_len_end_incl + 1, context_len_end_incl + 1 + query_len)
        )
        full_prompt = torch.cat(
            [fii.preamble_input_ids]
            + contexts
            + [fii.query_input_ids, fii.query_input_ids],
            dim=-1,
        )
        forward_result = model(
            full_prompt,
            output_attentions=AttentionReduction.LAYER_AND_HEAD,
        )
        reduced_attentions = (
            torch.stack(forward_result.attentions, dim=0).mean(dim=0).squeeze()
        )
        indices = rank_by_attention(
            branch_idx_to_document_node_idxs, query_node_idxs, reduced_attentions
        )
        print(f"\tattention_sort_step {i}: {indices}")
        _contexts_temp = [contexts[i] for i in indices]
        contexts = list(_contexts_temp)

        # Ensure we mirror the changes in the prompt string obj.
        _string_contexts_temp = [prompt_string_obj.contexts[i] for i in indices]
        prompt_string_obj.contexts = _string_contexts_temp

    return prompt_string_obj


def top_k_prompt_string(
    branch_weights: torch.Tensor,
    prompt_string_obj: SuperpositionPromptDef,
    forkjoin_topk: int = 1,
) -> SuperpositionPromptDef:
    prompt_string_obj = copy.deepcopy(prompt_string_obj)
    topk = torch.topk(branch_weights, largest=True, k=forkjoin_topk).indices.tolist()
    prompt_string_obj.contexts = [prompt_string_obj.contexts[i] for i in topk]
    return prompt_string_obj


def bm25_prefilter(
    prompt_string_obj: SuperpositionPromptDef,
    forkjoin_topk: int = 1,
    **kwargs,
) -> Tuple[SuperpositionPrompt, torch.Tensor]:
    prompt_string_obj = copy.deepcopy(prompt_string_obj)

    from rank_bm25 import BM25Okapi

    tokenized_documents = [document.split() for document in prompt_string_obj.contexts]
    tokenized_query = prompt_string_obj.query.split()

    # Create a BM25 model
    bm25 = BM25Okapi(tokenized_documents)

    # Get BM25 scores for each document
    bm25_scores = torch.tensor(list(bm25.get_scores(tokenized_query)))
    return top_k_prompt_string(
        bm25_scores, prompt_string_obj, forkjoin_topk=forkjoin_topk
    )


def tf_idf_prefilter(
    prompt_string_obj: SuperpositionPromptDef,
    forkjoin_topk: int = 1,
    **kwargs,
) -> Tuple[SuperpositionPrompt, torch.Tensor]:
    prompt_string_obj = copy.deepcopy(prompt_string_obj)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(prompt_string_obj.contexts)

    # Transform the query into TF-IDF representation
    query_tfidf = vectorizer.transform([prompt_string_obj.query])

    # Calculate cosine similarity between the query and documents
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    tf_idf_branch_weights = torch.tensor(cosine_similarities)
    return top_k_prompt_string(
        tf_idf_branch_weights, prompt_string_obj, forkjoin_topk=forkjoin_topk
    )


def contriever_prefilter(
    prompt_string_obj: SuperpositionPromptDef,
    forkjoin_topk: int = 1,
    **kwargs,
) -> Tuple[SuperpositionPrompt, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever")
    context_inputs = tokenizer(
        prompt_string_obj.contexts, padding=True, truncation=True, return_tensors="pt"
    )
    query_inputs = tokenizer(
        prompt_string_obj.query, padding=True, truncation=True, return_tensors="pt"
    )
    # Compute token embeddings

    def get_embeddings(inputs):
        outputs = model(**inputs)

        # Mean pooling
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            )
            return sentence_embeddings

        return mean_pooling(outputs[0], inputs["attention_mask"])

    context_embeddings = get_embeddings(context_inputs)
    query_embeddings = get_embeddings(query_inputs).squeeze()
    branch_weights = (context_embeddings * query_embeddings).sum(1)
    return top_k_prompt_string(
        branch_weights, prompt_string_obj, forkjoin_topk=forkjoin_topk
    )
