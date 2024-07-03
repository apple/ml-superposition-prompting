#!/usr/bin/env python3
import pathlib
from copy import deepcopy
from functools import partial
from typing import *

import torch
from pydantic.dataclasses import dataclass
from transformers.prompting.superposition import *

PROMPTS_ROOT = (pathlib.Path(__file__).parent / "prompts").resolve()

T = TypeVar("T")


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))


@dataclass
class SuperpositionPromptDef:
    preamble: str
    contexts: List[str]
    query: str
    task: str

    def get_forkjoin_input_ids(self, tokenizer, device) -> ForkjoinInputIds:
        def _tokenize(string: str) -> torch.Tensor:
            res = tokenizer(string, return_tensors="pt", padding=True).to(device)
            assert torch.all(res.attention_mask == 1)
            return res.input_ids

        def _tokenize_internal(string: str) -> torch.Tensor:
            input_ids = _tokenize(string)
            return input_ids[..., 1:]

        return ForkjoinInputIds(
            _tokenize(self.preamble),
            [_tokenize_internal(context) for context in self.contexts],
            _tokenize_internal(self.query),
            _tokenize_internal(self.task),
        )

    def to_prompt_cache(
        self, tokenizer, device, max_new_tokens, forkjoin_position_fn
    ) -> Tuple[ForkjoinDigraphs, SuperpositionPrompt, nx.DiGraph]:
        forkjoin_digraphs = prompt_cache_prompt_as_digraphs(
            **asdict(self.get_forkjoin_input_ids(tokenizer, device))
        )

        nested_digraph = prompt_cache_prompt_as_nested_digraph(
            forkjoin_digraphs, forkjoin_position_fn=forkjoin_position_fn
        )
        max_length = digraph_total_nodes(nested_digraph) + max_new_tokens + 1

        tp = digraph_to_superposition_prompt(
            nested_digraph,
            position_assignment_fn=get_position_ids_from_nested_digraph,
            causal_mask_fn=partial(
                compute_causal_mask_blockwise_recursive, propagate=False
            ),
            max_length=max_length,
            dtype=torch.float32,
        ).to(device=device)

        return forkjoin_digraphs, tp, nested_digraph

    def to_superposition_prompt(
        self, tokenizer, device, max_new_tokens, forkjoin_position_fn
    ) -> Tuple[ForkjoinDigraphs, SuperpositionPrompt, nx.DiGraph]:
        forkjoin_digraphs = forkjoin_prompt_as_digraphs(
            **asdict(self.get_forkjoin_input_ids(tokenizer, device))
        )

        nested_digraph = forkjoin_prompt_as_nested_digraph(
            forkjoin_digraphs, forkjoin_position_fn=forkjoin_position_fn
        )
        max_length = digraph_total_nodes(nested_digraph) + max_new_tokens + 1

        tp = digraph_to_superposition_prompt(
            nested_digraph,
            position_assignment_fn=get_position_ids_from_nested_digraph,
            causal_mask_fn=compute_causal_mask_blockwise_recursive,
            max_length=max_length,
            dtype=torch.float32,
        )
        return forkjoin_digraphs, tp, nested_digraph

    def format_llama_instruct_prompt(self):
        prompt = deepcopy(self)
        prompt.preamble = f"""[INST] <<SYS>>
{prompt.preamble.rstrip()}
<</SYS>>

"""
        prompt.query = prompt.query.rstrip()
        prompt.task = f" [/INST] Answer: "
        return prompt

    def format_instruct_prompt(self) -> "SuperpositionPromptDef":
        prompt = deepcopy(self)
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

        prompt.preamble = f"{INTRO_BLURB}\n\n{INSTRUCTION_KEY}\n{prompt.preamble}"
        prompt.task = f"\n{RESPONSE_KEY}\n{prompt.task}"
        return prompt

    def to_string(self) -> str:
        paths_str = "\n".join([c + self.query for c in self.contexts])
        return f"{self.preamble}{paths_str}{self.task}"

    @staticmethod
    def get_qa_prompt_string(
        question: str,
        documents: List[Union[Document, str]],
        mention_random_ordering: bool,
        query_aware_contextualization: bool,
    ) -> "SuperpositionPromptDef":
        if not question:
            raise ValueError(f"Provided `question` must be truthy, got: {question}")
        if not documents:
            raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

        if mention_random_ordering and query_aware_contextualization:
            raise ValueError(
                "Mentioning random ordering cannot be currently used with query aware contextualization"
            )

        if mention_random_ordering or query_aware_contextualization:
            assert False, "Not supported yet"

        # Format the documents into strings
        formatted_documents = []
        for document in documents:
            if isinstance(document, Document):
                formatted_documents.append(
                    f"[Document](Title: {document.title}) {document.text}"
                )
            else:
                assert isinstance(document, str)
                formatted_documents.append(f"[Document] {document}")

        return SuperpositionPromptDef(
            preamble="Write a high-quality answer for the given question using only the following relevant search results.\n\n",
            contexts=[d + "\n" for d in formatted_documents],
            query=f"\nQuestion: {question}\n",
            task="",
        )

    def flatten(self, superposition_flattening: int) -> "SuperpositionPromptDef":
        tps = copy.deepcopy(self)
        if superposition_flattening == 1:
            return tps
        assert superposition_flattening > 1
        formatted_documents = tps.contexts
        flattened_documents = []

        D = len(formatted_documents)
        assert D % superposition_flattening == 0
        branches = D // superposition_flattening
        for i in range(branches):
            start = i * superposition_flattening
            end = min(start + superposition_flattening, D)
            flattened_document = "".join(formatted_documents[start:end])
            flattened_documents.append(flattened_document)
        formatted_documents = flattened_documents
        return replace(tps, contexts=formatted_documents)


def get_closedbook_qa_prompt(question: str):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    with open(PROMPTS_ROOT / "closedbook_qa.prompt") as f:
        prompt_template = f.read().rstrip("\n")

    return prompt_template.format(question=question)
