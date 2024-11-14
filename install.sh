#!/bin/bash
# Elva
# Copyright (c) 2024-present NAVER Cloud Corp.
# MIT license

git submodule update --init

cp Elva/llava_init.py LLaVA/llava/__init__.py
cp Elva/model_init.py LLaVA/llava/model/__init__.py
cp Elva/llava_llama.py LLaVA/llava/model/language_model/llava_llama.py
cp Elva/llava_other_llms.py LLaVA/llava/model/language_model/llava_other_llms.py
cp Elva/conversation.py LLaVA/llava/conversation.py
cp Elva/mm_utils.py LLaVA/llava/mm_utils.py
cp Elva/encoder_builder.py LLaVA/llava/model/multimodal_encoder/builder.py
cp Elva/model_builder.py LLaVA/llava/model/builder.py
curl -o LLaVA/llava/model/language_model/modeling_openelm.py https://huggingface.co/apple/OpenELM-1_1B-Instruct/resolve/main/modeling_openelm.py
curl -o LLaVA/llava/model/language_model/configuration_openelm.py https://huggingface.co/apple/OpenELM-1_1B-Instruct/resolve/main/configuration_openelm.py

cp Elva/train_xformers.py LLaVA/llava/train/train_xformers.py
cp Elva/train.py LLaVA/llava/train/train.py

cp Elva/elva.py VLMEvalKit/vlmeval/vlm/elva.py
cp Elva/config.py VLMEvalKit/vlmeval/config.py
cp Elva/vlmevalkit_init.py VLMEvalKit/vlmeval/vlm/__init__.py
cp Elva/eval_gpt_review_parsing_bench.py LLaVA/llava/eval/eval_gpt_review_parsing_bench.py

pip uninstall -y llava vlmeval
pip install --upgrade --upgrade-strategy only-if-needed torch>=2.0.1 transformers>=4.42.4 protobuf<=3.20.1 moviepy validators xlsxwriter decord gradio omegaconf portalocker python-dotenv sty tiktoken timeout-decorator

cd LLaVA
pip install --no-deps -e .
cd ..
cd VLMEvalKit
pip install --no-deps -e .
cd ..
