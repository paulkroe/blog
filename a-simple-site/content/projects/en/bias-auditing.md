---
date: 2025-08-07T16:53:53-04:00
lastmod: 2025-08-07
showTableOfContents: false
tags: ["nlp", "ai-safety"]
title: "Ideological Bias Auditing in LLMs"
type: "post"
---

We introduce a lightweight way to **audit ideological steering in large language models** without needing model internals. The core idea: for a given sensitive topic (e.g., politics, religion), we periodically ask a fixed set of open-ended prompts, embed the model’s responses, and use a **permutation test on the cosine similarity of mean embeddings** to flag **distributional shifts**. If the distribution moves, that’s evidence the model’s behavior (potentially via system-prompt changes) has drifted.

This work was in part inspired an [incident](https://www.npr.org/2025/07/09/nx-s1-5462609/grok-elon-musk-antisemitic-racist-content) with **Grok’s system prompts and content moderation issues**, which highlighted the lack of external auditing tools for tracking subtle changes in model behavior over time.

Why this is useful:
- **Black-box friendly:** works with proprietary APIs—no weights or system prompts required.
- **Training-free & cheap:** sampling + embeddings + a simple statistical test.  
- **Practical signal:** catches even subtle framing shifts that humans might miss.

We validate the approach on three scenarios: (1) **religious framing**, (2) **subtle political manipulation via conspiracy framing**, and (3) a **real-world, long system prompt (Grok 4)**—showing reliable detection across models.

If you are interested feel free to read the [paper](https://arxiv.org/pdf/2509.12652).