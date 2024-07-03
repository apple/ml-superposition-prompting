import json
import random
from copy import deepcopy
from typing import *

import torch
from src.prompting import Document, SuperpositionPromptDef
from tqdm import tqdm
from xopen import xopen


def get_nq(
    input_path: str,
    use_random_ordering: bool = False,
    use_random_ordering_fix_gold: bool = True,
) -> Tuple:
    prompts: List[SuperpositionPromptDef] = []
    examples: List[Dict] = []
    all_model_documents: List[Document] = []

    with xopen(input_path) as fin:
        for i, line in enumerate(tqdm(fin)):
            torch.cuda.empty_cache()
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]

            # NOTE(tmerth): I added this...
            if not question.endswith("?"):
                question += "?"

            documents = []
            for ctx in deepcopy(input_example["ctxs"]):
                documents.append(Document.from_dict(ctx))
            if not documents:
                raise ValueError(
                    f"Did not find any documents for example: {input_example}"
                )

            if use_random_ordering and not use_random_ordering_fix_gold:
                random.shuffle(documents)
            elif use_random_ordering:
                assert use_random_ordering_fix_gold
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [
                    idx for idx, doc in enumerate(documents) if doc.isgold is True
                ]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            prompt_string = SuperpositionPromptDef.get_qa_prompt_string(
                question,
                documents,
                mention_random_ordering=False,
                query_aware_contextualization=False,
            )

            prompts.append(prompt_string)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)
    return prompts, examples, all_model_documents
