#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import dataclasses
import json
import logging
import math
import os
import pathlib
import pprint
import random
import subprocess
import warnings
from functools import partial
from typing import *

import torch
import yaml
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import AttentionReduction
from transformers.prompting.superposition import *

logger = logging.getLogger(__name__)


class Reasoning(BaseModel):
    id: int
    question: str
    answer: str
    paragraph_support_idx: Optional[int]


class Paragraph(BaseModel):
    idx: int
    title: str
    paragraph_text: str


class Example(BaseModel):
    id: str
    paragraphs: List[Paragraph]
    question: Optional[str]
    question_decomposition: List[Reasoning]
    answer: str
    answer_aliases: List[str]
    answerable: bool


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
    model,
    prompt_string_obj: "SuperpositionPromptDef",
    attention_sort_steps: int = 1,
    **kwargs,
) -> "SuperpositionPromptDef":
    fii = prompt_string_obj.get_forkjoin_input_ids(model.device)
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
        logger.info(f"\tattention_sort_step {i}: {indices}")
        _contexts_temp = [contexts[i] for i in indices]
        contexts = list(_contexts_temp)

        # Ensure we mirror the changes in the prompt string obj.
        _string_contexts_temp = [prompt_string_obj.contexts[i] for i in indices]
        prompt_string_obj.contexts = _string_contexts_temp

    return prompt_string_obj


def parameters_from_args(parser):
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration yaml file.",
    )
    config = parser.parse_args()
    with open(config.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    parameters = config["parameters"]
    logger.info(f"Loaded parameters:\n\t{pprint.pformat(parameters)}")
    return parameters


class EvalConfig(BaseModel):
    checkpoint_path: str
    data_dir: str
    output_dir: str

    # Superposition flattening is defined as `D / f`, where `f` is the
    # 'superposition factor' as defined by the paper.
    superposition_flattening: int = 1
    forkjoin_temperature: float = 0.01
    forkjoin_topk: Optional[int] = None
    topk_normalize: bool = True
    branch_weighting_fn: Literal[
        "bayesian_branch_weights",
        "attention_branch_weights",
        "random_branch_weights",
        "None",
    ] = "bayesian_branch_weights"
    format_alpaca: bool = True
    format_llama: bool = False
    preamble_str: str = "You are a question-answering assistant, who is careful to reference source material. Use the source(s) below to answer the user question.\n\n"
    task_str: str = ""
    max_new_tokens: int = 10
    generate_temperature: float = 1.0
    superposition_hops: int = 4
    pre_filter_fn: Literal[
        "bm25_prefilter",
        "tf_idf_prefilter",
        "contriever_prefilter",
        "partial(attention_sort, attention_sort_steps=1)",
        "partial(attention_sort, attention_sort_steps=2)",
        "partial(attention_sort, attention_sort_steps=3)",
        "None",
    ] = "None"
    superposition_prompt_creation_method: Literal[
        "to_superposition_prompt", "to_prompt_cache"
    ] = "to_superposition_prompt"


class Prediction(BaseModel):
    predicted_answerable: bool
    predicted_support_idxs: List[int]
    predicted_answer: str


class PredictionMetadata(BaseModel):
    selected_idxs: List[int]
    branch_weight_vectors: List[List[float]]
    full_prompt: str


@dataclasses.dataclass
class SuperpositionPromptDef:
    preamble: str
    contexts: List[str]
    query: str
    task: str

    def format_llama(self, eval_config: EvalConfig) -> "SuperpositionPromptDef":
        prompt = copy.deepcopy(self)
        prompt.preamble = f"""[INST] <<SYS>>
{prompt.preamble.rstrip()}
<</SYS>>

"""
        prompt.query = prompt.query.rstrip()
        prompt.task = f"{prompt.task} [/INST] Answer:"
        return prompt

    def format_alpaca(self, eval_config: EvalConfig) -> "SuperpositionPromptDef":
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        _preamble_str = (
            f"{INTRO_BLURB}\n\n{INSTRUCTION_KEY}\n{eval_config.preamble_str}"
        )
        _task_str = f"\n{RESPONSE_KEY}\n{eval_config.task_str}"
        return SuperpositionPromptDef(
            _preamble_str,
            self.contexts,
            self.query,
            task=_task_str,
        )

    @staticmethod
    def create(eval_config: EvalConfig, example: Example):
        base_prompt_string = SuperpositionPromptDef(
            preamble=eval_config.preamble_str,
            contexts=[
                f"[Document](Title: {paragraph.title}) {paragraph.paragraph_text}\n"
                for paragraph in example.paragraphs
            ],
            query=f"\nQuestion: {example.question}\n",
            task=eval_config.task_str,
        )
        if eval_config.format_alpaca:
            return base_prompt_string.format_alpaca(eval_config)
        elif eval_config.format_llama:
            return base_prompt_string.format_llama(eval_config)
        return base_prompt_string

    def get_forkjoin_input_ids(self, tokenizer, device):
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


def main():
    parser = argparse.ArgumentParser()
    parameters = parameters_from_args(parser)
    eval_config = EvalConfig(**parameters)
    logger.info(eval_config)

    assert (
        eval_config.branch_weighting_fn == "None" or eval_config.pre_filter_fn == "None"
    )
    examples = []

    with pathlib.Path(eval_config.data_dir).open("r") as fp:
        lines = fp.readlines()

        for line in lines:
            example = Example(**json.loads(line))

            random.shuffle(example.paragraphs)
            examples.append(example)

    logger.info(f"Num examples: {len(examples)}")

    torch.set_grad_enabled(False)
    f16_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if "meta-llama" in eval_config.checkpoint_path:
        model = (
            AutoModelForCausalLM.from_pretrained(str(eval_config.model_name), local_files_only=True)
            .to(dtype=f16_dtype)
            .cuda()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(str(eval_config.model_name))
        tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    else:
        tokenizer = AutoTokenizer.from_pretrained(eval_config.checkpoint_path)
        model = (
            AutoModelForCausalLM.from_pretrained(eval_config.checkpoint_path)
            .to(dtype=f16_dtype)
            .cuda()
            .eval()
        )

    logger.info(eval_config.branch_weighting_fn)
    branch_weighting_fn = eval(eval_config.branch_weighting_fn)

    predictions = []
    prediction_metadatas = []
    eval_config.pre_filter_fn = eval(eval_config.pre_filter_fn)

    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        prompt_string = SuperpositionPromptDef.create(eval_config, example)
        if eval_config.pre_filter_fn is not None:
            prompt_string = eval_config.pre_filter_fn(model, prompt_string)

        logger.info(f"idx {idx}: {len(prompt_string.contexts)} paragraph(s)")
        if eval_config.superposition_flattening > 1:
            warnings.warn(
                "Be careful of metrics which measure document citing--this logic will interfere with the indexing."
            )
            flattened_contexts = []

            D = len(prompt_string.contexts)
            branches = int(math.ceil(D / eval_config.superposition_flattening))
            for i in range(branches):
                start = i * eval_config.superposition_flattening
                end = min(start + eval_config.superposition_flattening, D)
                flattened_context = "".join(prompt_string.contexts[start:end])
                flattened_contexts.append(flattened_context)

            prompt_string.contexts = flattened_contexts

        forkjoin_input_ids = prompt_string.get_forkjoin_input_ids(
            device=model.device, tokenizer=tokenizer
        )

        prefix_position_ids_to_overwrite = None
        selected_idx = None

        original_idxs = list(range(len(prompt_string.contexts)))
        selected_idxs = []
        branch_weight_vectors = []

        if eval_config.superposition_hops > 0:
            assert eval_config.branch_weighting_fn is not None
            assert (
                eval_config.superposition_prompt_creation_method
                == "to_superposition_prompt"
            )
            hop_iters = list(range(eval_config.superposition_hops))
            for t_hop in hop_iters:
                forkjoin_digraphs = forkjoin_input_ids.to_forkjoin_digraphs()
                nested_digraph = forkjoin_prompt_as_nested_digraph(forkjoin_digraphs)
                max_length = (
                    digraph_total_nodes(nested_digraph) + eval_config.max_new_tokens + 1
                )
                logger.info(f"max_length: {max_length}")

                superposition_prompt = digraph_to_superposition_prompt(
                    nested_digraph,
                    position_assignment_fn=get_position_ids_from_nested_digraph,
                    causal_mask_fn=compute_causal_mask_blockwise_recursive,
                    max_length=max_length,
                    dtype=f16_dtype,
                ).to(device=model.device)
                assert superposition_prompt.attention_logit_biases.dtype == f16_dtype

                # `prefix_position_ids_to_overwrite` should only be None when t_hop is zero.
                assert (prefix_position_ids_to_overwrite is None) == (t_hop == 0)
                if prefix_position_ids_to_overwrite is not None:
                    superposition_prompt = (
                        update_superposition_prompt_with_position_id_range(
                            superposition_prompt, prefix_position_ids_to_overwrite
                        )
                    )

                assert eval_config.forkjoin_temperature > 0.0

                extra_forward_kwargs = dict(
                    output_attentions=AttentionReduction.LAYER_AND_HEAD
                    if branch_weighting_fn == attention_branch_weights
                    else False
                )
                forward_result = model.superposition_forward(
                    superposition_prompt, use_cache=False, **extra_forward_kwargs
                )

                branch_weight_kwargs = dict()
                if branch_weighting_fn == bayesian_branch_weights:
                    branch_weight_kwargs.update(
                        dict(
                            logits=forward_result.logits,
                        )
                    )
                elif branch_weighting_fn == attention_branch_weights:
                    assert forward_result.attentions is not None
                    attention_probs = (
                        torch.stack(forward_result.attentions, dim=0)
                        .mean(dim=0)
                        .squeeze()
                    )
                    branch_weight_kwargs.update(
                        dict(
                            reduced_attentions=attention_probs,  # Need to strip degenerate batch_dim
                        )
                    )
                elif branch_weighting_fn == random_branch_weights:
                    pass
                else:
                    assert False

                branch_weights = branch_weighting_fn(
                    forkjoin_digraphs=forkjoin_digraphs,
                    **branch_weight_kwargs,
                )

                if not isinstance(branch_weights, torch.Tensor):
                    assert isinstance(branch_weights, ForkjoinLogLikelihoods)
                    branch_weights = branch_weights.branches

                branch_weight_vectors.append(branch_weights.squeeze().tolist())
                selected_idx = (
                    torch.topk(branch_weights.squeeze(), k=1).indices.squeeze().item()
                )
                prefix_position_ids_to_overwrite = (
                    prefix_position_ids_from_branch_selections(
                        forkjoin_digraphs,
                        superposition_prompt,
                        selection_idxs=[selected_idx],
                    )
                )
                selected_idxs.append(original_idxs.pop(selected_idx))
                forkjoin_input_ids = push_branch_to_preamble(
                    forkjoin_input_ids, selected_idx
                )
                assert len(forkjoin_input_ids.context_input_ids_list) > 0

        if eval_config.superposition_hops == 0:
            forkjoin_digraphs = forkjoin_input_ids.to_forkjoin_digraphs()

        if eval_config.forkjoin_topk is not None:
            assert (
                eval_config.superposition_prompt_creation_method
                == "to_superposition_prompt"
            )

        if (
            eval_config.superposition_prompt_creation_method
            == "to_superposition_prompt"
        ):
            forkjoin_digraphs = forkjoin_input_ids.to_forkjoin_digraphs()
            nested_digraph = forkjoin_prompt_as_nested_digraph(forkjoin_digraphs)
            max_length = (
                digraph_total_nodes(nested_digraph) + eval_config.max_new_tokens + 1
            )
            logger.info(f"max_length: {max_length}")
            superposition_prompt = digraph_to_superposition_prompt(
                nested_digraph,
                position_assignment_fn=get_position_ids_from_nested_digraph,
                causal_mask_fn=compute_causal_mask_blockwise_recursive,
                max_length=max_length,
                dtype=f16_dtype,
            ).to(device=model.device)

            if eval_config.forkjoin_topk is not None:
                if branch_weighting_fn == bayesian_branch_weights:
                    attenuate_fn = bayesian_attenuate_forkjoin_branches
                elif branch_weighting_fn == attention_branch_weights:
                    attenuate_fn = attention_attenuate_forkjoin_branches
                else:
                    assert False, branch_weighting_fn

                extra_forward_kwargs = dict(
                    output_attentions=AttentionReduction.LAYER_AND_HEAD
                    if attenuate_fn == attention_attenuate_forkjoin_branches
                    else False
                )
                forward_result = model.superposition_forward(
                    superposition_prompt,
                    shift_labels=False,
                    use_cache=False,
                    **extra_forward_kwargs,
                )

                attenuate_fn_kwargs = dict()
                if branch_weighting_fn == bayesian_branch_weights:
                    attenuate_fn_kwargs.update(
                        dict(
                            logits=forward_result.logits,
                        )
                    )
                elif branch_weighting_fn == attention_branch_weights:
                    attention_probs = (
                        torch.stack(forward_result.attentions, dim=0)
                        .mean(dim=0)
                        .squeeze()
                    )
                    attenuate_fn_kwargs.update(
                        dict(
                            reduced_attentions=attention_probs,
                        )
                    )
                else:
                    assert False

                superposition_prompt, branch_weight_tensor = attenuate_fn(
                    forkjoin_digraphs=forkjoin_digraphs,
                    superposition_prompt=superposition_prompt,
                    forkjoin_temperature=None,
                    forkjoin_topk=eval_config.forkjoin_topk,
                    topk_normalize=eval_config.topk_normalize,
                    **attenuate_fn_kwargs,
                )
        elif eval_config.superposition_prompt_creation_method == "to_prompt_cache":
            forkjoin_digraphs = prompt_cache_prompt_as_digraphs(
                **asdict(forkjoin_input_ids)
            )
            nested_digraph = prompt_cache_prompt_as_nested_digraph(forkjoin_digraphs)
            max_length = (
                digraph_total_nodes(nested_digraph) + eval_config.max_new_tokens + 1
            )
            logger.info(f"max_length: {max_length}")
            superposition_prompt = digraph_to_superposition_prompt(
                nested_digraph,
                position_assignment_fn=get_position_ids_from_nested_digraph,
                causal_mask_fn=partial(
                    compute_causal_mask_blockwise_recursive, propagate=False
                ),
                max_length=max_length,
                dtype=f16_dtype,
            ).to(device=model.device)
        else:
            assert False, eval_config.superposition_prompt_creation_method

        assert superposition_prompt.attention_logit_biases.dtype == f16_dtype

        assert (prefix_position_ids_to_overwrite is not None) == (
            eval_config.superposition_hops > 0
        )
        if prefix_position_ids_to_overwrite is not None:
            assert (
                eval_config.superposition_prompt_creation_method
                == "to_superposition_prompt"
            )
            superposition_prompt = update_superposition_prompt_with_position_id_range(
                superposition_prompt, prefix_position_ids_to_overwrite
            )

        outputs = model.superposition_prompt_generate(
            superposition_prompt,
            max_new_tokens=eval_config.max_new_tokens,
            temperature=eval_config.generate_temperature,
            use_cache=True,
            past_key_values=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=False,
        ).squeeze()

        response_output_ids = outputs[superposition_prompt.input_ids.numel() :]
        response = tokenizer.decode(
            response_output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        full_prompt = tokenizer.decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.info(full_prompt)
        logger.info(f"Acceptable answers:\t{[example.answer] + example.answer_aliases}")

        prediction = Prediction(
            predicted_answer=response,
            predicted_support_idxs=[2],
            predicted_answerable=True,
        )
        prediction_metadata = PredictionMetadata(
            selected_idxs=selected_idxs,
            branch_weight_vectors=branch_weight_vectors,
            full_prompt=full_prompt,
        )
        predictions.append(prediction)
        prediction_metadatas.append(prediction_metadata)

        del superposition_prompt
        del forkjoin_digraphs
        del nested_digraph

    output_dir_path = pathlib.Path(eval_config.output_dir)
    output_dir_path.mkdir(exist_ok=True)
    output_file = output_dir_path / "responses.jsonl"
    metrics_file = output_dir_path / "metrics.jsonl"

    with output_file.open("w") as fp:
        for data_objs in zip(examples, predictions, prediction_metadatas):
            json_objs = [x.model_dump(mode="json") for x in data_objs]
            output_json = dict()
            for json_obj in json_objs:
                output_json.update(json_obj)
            assert len(output_json) == sum(len(x) for x in json_objs)
            json.dump(output_json, fp)
            fp.write("\n")

    env = {}
    env.update(os.environ)
    cmd = f"poetry run python third_party/StonyBrookNLP/musique/evaluate_v1.0.py {output_file} {eval_config.data_dir} --output_filepath {metrics_file}"
    logger.info(subprocess.run(cmd.split(" "), env=env))
    with metrics_file.open("r") as fp:
        logger.info(fp.read())


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
