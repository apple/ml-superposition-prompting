from typing import *

from pydantic import BaseModel, ConfigDict


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Path to data with questions and documents to use.
    input_path: str

    # Model to use in generating responses.
    model_name: str

    # Temperature to use in generation.
    temperature: float = 0.0

    # Top-p to use in generation.
    top_p: int = 1

    batch_size: int = 1

    # Run the model in closed-book mode (i.e., don't use documents).
    closedbook: bool = False

    # Mention that search results are ordered randomly in the prompt.
    prompt_mention_random_ordering: bool = False

    # Randomize the ordering of the distractors, rather than sorting by relevance.
    use_random_ordering: bool = False

    # Place the question both before and after the documents.
    query_aware_contextualization: bool = False

    num_gpus: int = 1

    # Maximum memory to use per GPU (in GiB) for multi-device parallelism, e.g., 80.
    max_memory_per_gpu: int = 80

    # Path to write output file of generated response.
    output_responses_path: str

    # Maximum number of new tokens to generate.
    max_new_tokens: int = 100

    superposition: bool = True

    # Superposition flattening is defined as `D / f`, where `f` is the
    # 'superposition factor' as defined by the paper and `D` is the number of
    # documents.
    superposition_flattening: int

    forkjoin_temperature: Optional[float] = 0.01

    forkjoin_topk: Optional[int] = None

    attenuate_fn: Union[
        Literal[
            "bayesian_attenuate_forkjoin_branches",
            "attention_attenuate_forkjoin_branches",
            "None",
        ],
        Callable,
    ]

    forkjoin_position_fn: Union[
        Literal["equilibrium_from_forkjoin_digraphs", "max_from_forkjoin_digraphs"],
        Callable,
    ] = "equilibrium_from_forkjoin_digraphs"

    # Path to write output file of generated response.
    output_score_path: str

    use_cache: bool = False

    pre_filter_fn: Union[
        Literal[
            "bm25_prefilter",
            "tf_idf_prefilter",
            "contriever_prefilter",
            "partial(attention_sort, attention_sort_steps=1)",
            "partial(attention_sort, attention_sort_steps=2)",
            "partial(attention_sort, attention_sort_steps=3)",
            "None",
        ],
        Callable,
    ] = "None"

    superposition_prompt_creation_fn: Union[
        Literal["to_superposition_prompt", "to_prompt_cache"],
        Callable,
    ] = "to_superposition_prompt"

    topk_normalize: bool = False

    use_random_ordering_fix_gold: bool = True

    pos_round: bool = False
