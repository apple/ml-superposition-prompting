#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run Bloom to get responses.

Currently supports `mosaicml/mpt-30b-instruct` and `mosaicml/mpt-30b`.

The retrieval results are used in the exact order that they're given.
"""

import argparse
import dataclasses
import json
import logging
import math
import pathlib
import pprint
import random
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import *

import torch
import yaml
from src.config import (
    EvalConfig,
)
from src.datasets import get_nq
from src.evaluate import main as evaluate_main
from src.prefilter import *
from src.prompting import (
    Document,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import AttentionReduction
from transformers.prompting.superposition import *
from xopen import xopen

logger = logging.getLogger(__name__)
random.seed(0)


def inference_main(config):
    assert (
        not config.use_cache
    ), "Caching for generation doesn't seem to work with bloom"
    assert config.num_gpus == 1
    assert config.batch_size == 1
    # Create directory for output path if it doesn't exist.
    output_path_obj = pathlib.Path(config.output_responses_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    assert not output_path_obj.exists(), f"Already exists: {output_path_obj}"

    _model = config.model_name.lower()

    if any(x in _model for x in {"bloomz", "bigscience"}):
        logger.warning(
            f"Model {config.model_name} appears to be an instruct model, applying instruct formatting"
        )
    elif "meta-llama" in _model:
        logger.warning(
            f"Model {config.model_name} appears to be an instruct model, applying instruct formatting"
        )
    elif "mpt" in _model:
        if "instruct" not in _model:
            logger.warning(
                f"Model {config.model_name} appears to be an instruct model, applying instruct formatting"
            )
        else:
            logger.warning(
                f"Model {config.model_name} does not appear to be an instruct model, so not applying instruct formatting"
            )
    else:
        assert False

    assert not config.closedbook
    assert config.superposition
    prompts, examples, all_model_documents = get_nq(
        config.input_path,
        config.use_random_ordering,
        config.use_random_ordering_fix_gold,
    )

    # Fetch all of the prompts
    if any(x in _model for x in {"bloomz", "bigscience"}):
        prompts = [prompt_string.format_instruct_prompt() for prompt_string in prompts]
    elif "mpt" in _model:
        assert "instruct" in _model
        prompts = [prompt_string.format_instruct_prompt() for prompt_string in prompts]
    elif "meta-llama" in _model:
        prompts = [
            prompt_string.format_llama_instruct_prompt() for prompt_string in prompts
        ]
    elif "open_llama" in _model:
        pass
    else:
        assert False

    # Get responses for all of the prompts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError(
            "Unable to find CUDA device with torch. Please use a CUDA device to run this script."
        )

    logger.info("Loading model and tokenizer...")
    if "meta-llama" in config.model_name:
        model = (
            AutoModelForCausalLM.from_pretrained(str(config.model_name), local_files_only=True)
            .to(torch.bfloat16)
            .cuda()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(str(config.model_name))
        tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    else:
        extra_config_kwargs = {}
        extra_tokenizer_kwargs = {}
        if "mpt" in config.model_name:
            extra_config_kwargs["max_seq_len"] = 5120
            extra_tokenizer_kwargs["model_max_length"] = 5120
        model_config = AutoConfig.from_pretrained(
            config.model_name, **extra_config_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, **extra_tokenizer_kwargs
        )
        tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
            torch_dtype=torch.float16,
        )

    model = model.to(device)
    logger.info(f"...loaded model: {model}")

    do_sample = config.temperature > 0.0
    assert config.batch_size == 1

    with torch.autocast(device, dtype=torch.float16), torch.no_grad():
        for i, prompt_def in tqdm(enumerate(prompts), total=math.ceil(len(prompts))):
            torch.cuda.empty_cache()

            branch_weight_vector = None
            if config.pre_filter_fn is not None:
                prompt_def = config.pre_filter_fn(
                    tokenizer=tokenizer,
                    model=model,
                    prompt_string_obj=prompt_def,
                    forkjoin_topk=config.forkjoin_topk,
                )
                if config.forkjoin_topk > 1:
                    # Ensure that the prompt is flattened.
                    assert config.superposition_flattening == 1
                    prompt_def.flatten(superposition_flattening=config.forkjoin_topk)

            prompt_def = prompt_def.flatten(config.superposition_flattening)

            assert config.superposition

            method = getattr(prompt_def, config.superposition_prompt_creation_fn)
            forkjoin_digraphs, superposition_prompt, nested_digraph = method(
                tokenizer,
                device,
                config.max_new_tokens,
                config.forkjoin_position_fn,
            )
            if config.pos_round:
                superposition_prompt.position_ids = torch.round(
                    superposition_prompt.position_ids
                )

            if config.attenuate_fn is not None:
                assert (
                    config.superposition_prompt_creation_fn
                    == "to_superposition_prompt"
                )
                assert (
                    config.forkjoin_temperature is not None
                    and config.forkjoin_temperature > 0.0
                ) or (config.forkjoin_topk is not None)
                extra_forward_kwargs = dict(
                    output_attentions=AttentionReduction.LAYER_AND_HEAD
                    if config.attenuate_fn == attention_attenuate_forkjoin_branches
                    else False
                )
                forward_result = model.superposition_forward(
                    superposition_prompt,
                    shift_labels=False,
                    use_cache=False,
                    **extra_forward_kwargs,
                )

                attenuate_fn_kwargs = dict()
                if config.attenuate_fn in {
                    bayesian_attenuate_forkjoin_branches,
                    bayesian_attenuate_forkjoin_branches,
                }:
                    attenuate_fn_kwargs.update(
                        dict(
                            logits=forward_result.logits,
                        )
                    )
                elif config.attenuate_fn == attention_attenuate_forkjoin_branches:
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

                superposition_prompt, branch_weight_tensor = config.attenuate_fn(
                    forkjoin_digraphs=forkjoin_digraphs,
                    superposition_prompt=superposition_prompt,
                    forkjoin_temperature=config.forkjoin_temperature,
                    forkjoin_topk=config.forkjoin_topk,
                    topk_normalize=config.topk_normalize,
                    **attenuate_fn_kwargs,
                )
                branch_weight_tensor = (
                    branch_weight_tensor
                    if isinstance(branch_weight_tensor, torch.Tensor)
                    else branch_weight_tensor.branches
                )
                branch_weight_vector = branch_weight_tensor.squeeze().tolist()
            else:
                forward_result = model.superposition_forward(
                    superposition_prompt,
                    shift_labels=False,
                    use_cache=False,
                )

            past_key_values = (
                forward_result.past_key_values if config.use_cache else None
            )

            total_nodes = digraph_total_nodes(nested_digraph)
            logger.info(f"{i}:\ttotal_nodes: {total_nodes}")

            outputs = model.superposition_prompt_generate(
                superposition_prompt,
                max_new_tokens=config.max_new_tokens,
                do_sample=do_sample,
                temperature=config.temperature if do_sample else None,
                top_p=config.top_p if do_sample else None,
                use_cache=True,
                past_key_values=past_key_values,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=False,
            )
            inputs = superposition_prompt.input_ids
            assert len(outputs) == 1, "Only batch size of 1 is supported"
            assert inputs.ndim == 2
            assert len(inputs) == len(outputs)

            input_ids = inputs[0]
            generated_sequence = outputs[0]

            responses = []
            assert input_ids.ndim == 1

            text = tokenizer.decode(generated_sequence)
            logger.info(text)
            logger.info(f"Acceptable: {examples[i]['answers']}")

            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    tokenizer.decode(
                        input_ids,
                        # skip_special_tokens=True,
                        # clean_up_tokenization_spaces=True,
                    )
                )
            full_prompt = text[:prompt_length]
            new_text = text[prompt_length:]
            response = new_text
            responses.append(new_text)

            with xopen(config.output_responses_path, "a") as f:
                assert len(outputs) == 1, "Only batch size of 1 is supported"
                # for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
                example = examples[i]
                model_documents = all_model_documents[i]
                prompt = prompts[i]

                assert len(responses) == 1
                response = responses[0]

                output_example = deepcopy(example)
                # Add some extra metadata to the output example
                output_example["model_prompt"] = dataclasses.asdict(prompt)
                output_example["model_documents"] = [
                    dataclasses.asdict(document)
                    if isinstance(document, Document)
                    else document
                    for document in model_documents
                ]
                output_example["model_answer"] = response
                output_example["model"] = config.model_name
                output_example["model_temperature"] = config.temperature
                output_example["model_top_p"] = config.top_p
                output_example["model_prompt_mention_random_ordering"] = (
                    config.prompt_mention_random_ordering
                )
                output_example["model_use_random_ordering"] = config.use_random_ordering
                output_example["final_prompt"] = full_prompt
                output_example["branch_weight_vector"] = branch_weight_vector
                f.write(json.dumps(output_example) + "\n")


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


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parameters = parameters_from_args(parser)
    config = EvalConfig(**parameters)
    for key in {"attenuate_fn", "pre_filter_fn", "forkjoin_position_fn"}:
        setattr(config, key, eval(getattr(config, key)))

    inference_main(config)
    logger.info("finished running %s", sys.argv[0])

    evaluate_main(
        config.output_responses_path,
        config.output_score_path,
    )
