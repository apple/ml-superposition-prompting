#!/usr/bin/env bash

# Core and library.
pytest --noconftest third_party/huggingface/transformers/tests/prompting/superposition/test_superposition_prompt.py

# Model-specific.
pytest --noconftest third_party/huggingface/transformers/tests/models/bloom/test_modeling_bloom.py -k 'superposition'
pytest --noconftest third_party/huggingface/transformers/tests/models/mpt/test_modeling_mpt.py -k 'superposition'
pytest --noconftest third_party/huggingface/transformers/tests/models/llama/test_modeling_llama.py -k 'superposition'