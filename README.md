<div align="center">

# Elva: Efficient Language and Vision Assistant
[![Paper](https://img.shields.io/badge/Paper-arxiv.2406.11823-red)](https://arxiv.org/abs/2406.11823)
[![Conference](https://img.shields.io/badge/EMNLP-2024-blue)](#how-to-cite)

Official Implementation of Elva | [Paper](https://arxiv.org/abs/2406.11823) | [Slide](https://docs.google.com/presentation/d/1SfPHqJs7v38_IvNDcvrWnhIQU2AjzXxHmCdflz_wDaw/edit?usp=sharing) | [Poster](https://docs.google.com/presentation/d/1FiRnXL4sp3rBoHdYMQYt4PruFrNR1x5EjGQye_uB3Jk/edit?usp=sharing)

<img width="800" alt="teaser" src="https://github.com/user-attachments/assets/2ebe381e-7a3c-4a0d-b299-2e2a216f1290">

</div>

## Introduction

Welcome to the official repository for Elva, a language and vision assistant developed with the goal of enhancing efficiency in vision-language models. Recent advancements in Large Vision-Language Models (LVLMs) have improved visually-situated language understanding but often come with higher computational demands. In response, Elva offers a streamlined and faster alternative.

Our work focuses on optimizing key components to balance performance with manageable computational costs. Through this repository, we hope to offer our findings and techniques as helpful resources for those interested in the field. By refining dataset strategies, enhancing vision module designs, and optimizing supervision methods, Elva makes noticeable improvements in speed and accuracy.

Here, you'll find the resources and codebase needed to replicate our findings and explore further possibilities. Our experiments, ranging from 160M to 13B parameters, provide insights we believe can benefit ongoing work in visually-situated natural language understanding.

For more detailed information, please refer to our paper linked below. We hope you find this repository useful and look forward to seeing how it contributes to your research and applications:<br>
> [**On Efficient Language and Vision Assistants for Visually-Situated Natural Language Understanding: What Matters in Reading and Reasoning**](https://arxiv.org/abs/2406.11823).<br>
> [Geewook Kim](https://geewook.kim) and [Minjoon Seo](https://scholar.google.com/citations?user=zYze5fIAAAAJ). In EMNLP 2024.

## Updates

**_2024-11-14_** Initial release of the codebase.

## Installation

To get started with Elva, execute the following installation command:

```bash
bash install.sh
```

## Pretrained Models

| Model               | Link |
|---------------------|------|
| Elva-LLaMA-160M     | [gwkrsrch/elva-llama-160m](https://huggingface.co/gwkrsrch/elva-llama-160m) |
| Elva-TinyVicuna-1B  | Coming soon! |
| Elva-Phi3-3.8B      | Coming soon! |
| Elva-Vicuna-7B      | Coming soon! |
| Elva-Vicuna-13B     | Coming soon! |

You can start by experimenting with our smallest 0.2B model. More models will be available soon. Also, you can train your own models using the instructions in the next sections.

## Testing Pretrained Models

To test the pretrained models, use this command:

```bash
cd VLMEvalKit
torchrun --nproc-per-node=1 run.py --data ScienceQA_TEST --model elva-llama-160m
```

This will show you the test results.

## Training Your Own Models

You can train your own models with the provided scripts:
- `train_llama.sh`
- `train_openelm.sh`
- `train_phi3.sh`

## How to Cite

If you find Elva helpful for your work, please include the following citation:

```bibtex
@inproceedings{kim-seo-2024-efficient,
    title = "On Efficient Language and Vision Assistants for Visually-Situated Natural Language Understanding: What Matters in Reading and Reasoning",
    author = "Kim, Geewook  and
      Seo, Minjoon",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.944",
    pages = "16978--17000",
}
```

## License
```
Elva
Copyright (c) 2024-present NAVER Cloud Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
