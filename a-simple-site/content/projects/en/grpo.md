---
date: 2025-09-22T16:53:53-04:00
# description: ""
# image: ""
lastmod: 2025-09-22
showTableOfContents: false
tags: ["nlp", "rl"]
title: "LLM post-training with GRPO"
type: "post"
---
Inspired by the [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment/blob/main/nano_r1.ipynb) project, I created a minimal, easy-to-understand implementation of Group Relative Policy Optimization (GRPO) for post-training large language models.

This project fine-tunes Qwen2.5 3B for reasoning tasks, focusing on simplicity and clarity rather than maximal performance.  
Part of this is an independent Jupyter notebook that runs on an 80GB A100 GPU, producing interesting results in under an hour and offering a clear, from-scratch introduction to GRPO and RL for LLMs.

Check out the code [here](https://github.com/paulkroe/llm-rl).

I also wrote a brief [blog post](/posts/en/grpo/) introducing RL for LLMs and walking through GRPO step by step.  