---
date: 2025-09-10T16:53:53-04:00
# description: "From-scratch implementation of Double DQN with prioritized replay to solve Atari Breakout."
# image: "/images/dqn_breakout.png"
lastmod: 2025-09-10
showTableOfContents: false
tags: ["nlp", "rl"]
title: "DQN for Atari Breakout from Scratch"
type: "post"
---

![DQN agent playing Atari Breakout](/gifs/dqn_epoch_500.gif)

**Deep Q-Networks (DQNs)** were a milestone in deep reinforcement learning, famously solving Atari games directly from pixels.  
For this project, I built a **from-scratch implementation** of key RL techniques to train an agent to play **Atari Breakout**, including:

- **Deep Q-Learning** – iconic approach for learning Atari games directly from pixels
- **Double Q-Learning** – reducing overestimation bias in Q-learning  
- **Prioritized Replay** – sampling important experiences more frequently

I noticed there are surprisingly few clear explanations of how to implement these ideas step-by-step.
Thus, I tried to write something useful and if you are interested, feel free to read the blog post [here](/posts/en/dqn-atari/).

You can also explore the code on [GitHub](https://github.com/paulkroe/atari-rl).

### References
Here are the main papers that inspired this project:
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/pdf/1312.5602)  
- [Double Q-learning (van Hasselt et al., 2015)](https://arxiv.org/pdf/1509.06461)  
- [Prioritized Experience Replay (Schaul et al., 2016)](https://arxiv.org/pdf/1511.05952)  

