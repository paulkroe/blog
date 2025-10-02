---
date: 2025-09-10T16:53:53-04:00
# description: ""
# image: ""
lastmod: 2025-09-10
showTableOfContents: false
tags: ["rl", "work-in-progress"]
title: "DQN for Atari Breakout from Scratch"
type: "post"
---

I’ve always had a lurking interest in reinforcement learning, but only recently did I find the time to revisit some of the basics. Part of “going back to the roots” was looking at the first truly groundbreaking deep reinforcement learning papers. Among these, the ones that showed deep RL agents could play Atari games. Back in 2015, this was so astonishing that it even made [mainstream news](https://www.nbcnews.com/tech/innovation/ai-learned-atari-games-human-now-it-beats-them-n312206).

Besides the fact that the original papers are freely available ([this list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) is a great place to start), there’s no shortage of tutorials, blog posts, and lecture notes explaining how deep RL works. Still, I thought it would be fun to write one more “from scratch” implementation—partly for my own learning and partly because I think I can add a few fresh angles that others might find useful. Feedback is, of course, very welcome.

If you’re mostly here for the **code**, feel free to jump ahead to the **implementation section**. The first couple of sections are just a recap of the key ideas we’ll need.

# The Problem

Imagine you’re as terrible at arcade games as I am. You still want to impress your friends with a massive high score, so you decide to train a computer to play for you. Ideally, you want the computer to have the same inputs that you do, i.e. just the raw screen pixels.

Fortunately, Mnih et al. solved just this problem in their influential 2015 paper, [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602). Their big idea was simple but powerful: feed the current game frame into a neural network, and have it predict which action is best.

Of course, the real question is: how does the network know which action is good in the first place? That’s where reinforcement learning comes in. We’ll recap just enough theory to make sense of this before diving into the machine deep machine learning version.

# A (very[]()) brief Introduction to Reinforcement Learning

Reinforcement learning is about training an agent to make decisions by interacting with an environment. At each step, the agent sees a **state** (s), chooses an **action** (a), receives a **reward** (r), and transitions to a new state (s'). This loop is often formalized as a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process).

The agent’s goal is to maximize the expected **return**, which is the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}
$$

where $0 \leq \gamma < 1$ is the discount factor that balances immediate versus long-term rewards. The agent’s behavior is governed by a **policy** $\pi$, which tells us how actions are chosen in each state. Formally, $\pi(a \mid s)$ is the probability of taking action $a$ in state $s$.

To evaluate policies, we define **value functions**. The state-value function measures the expected return from state $s$:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]
$$

The action-value function (or **Q-function**) measures the expected return from taking action $a$ in state $s$:

$$
Q^\pi(s,a) = \mathbb{E}_\pi \big[ G_t \mid S_t = s, A_t = a \big]
$$

The optimal Q-function satisfies the Bellman optimality equation:

$$
Q^{\ast}(s,a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s',a') \mid s,a \right]
$$

Interestingly, (due to the uniqueness of fixed points[]()) the optimal Q-function is unique. That means that for any Q-function $Q$ satisfing the above equation, we have $Q = Q^{\ast}$

Building open this result, Q-learning is an algorithm that directly approximates the optimal Q-function. It updates estimates using:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

Here $\alpha$ is the learning rate, and the term in brackets is the **temporal-difference error**. Once trained, the agent acts greedily:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

# Deep Q-Learning

Now let’s return to the Atari problem. Using the classical RL formulation, the idea behind deep Q-learning (DQN[]()) is actually quite straightforward. Suppose our environment has $n_a$ possible actions. We take a neural network, feed it the current screen frame as input, and let it output an $n_a$-dimensional vector. For a given state $s$, the $i$-th entry corresponds to the Q-value $Q(s, a_i)$. By taking the argmax over this vector, we select the action the network thinks is best.

During training, we execute that action in the environment, observe a reward $r$ and a next state $s'$, and then update the network to reduce the temporal-difference error defined by the Bellman equation. Concretely, the loss is based on the difference between the predicted Q-value and the target value. To make this stable in practice, the authors introduce a **target network**, a copy of the Q-network whose parameters are held fixed for a while and only updated periodically. This trick prevents the network from chasing its own moving targets and was one of the key innovations of the original DQN paper.

$$
y = r + \gamma \max_{a'} Q(s',a'; \theta^-)
$$

$$
L(\theta) = \big( Q(s,a;\theta) - y \big)^2
$$

Here $\theta$ are the network parameters, and $\theta^-$ are the parameters of the target network.

While there are many additional details needed to make this work well in practice (we’ll get to those later[]()), there’s one 
conceptual issue to address right away: using only a single frame as input often makes the state of Atari games ambiguous. For example, looking at the [screenshot](https://en.wikipedia.org/wiki/Breakout_(video_game)) below, you can’t immediately tell which way the ball is moving.

<img src="/images/breakout2600.png" alt="atari breakout frame" width="500"/>  

Humans solve this by using information about previously seen frames to infer the direction in which the ball is moving. But our model, in its current form, has no explicit memory of past states. To give it a similar capability, the original DQN stacked the last four frames together as input to their network. This way, the network can infer motion and dynamics across time. (Later work also experimented with adding [recurrent networks like LSTMs](https://arxiv.org/pdf/1507.06527) to give agents true memory, but we’ll stick to the frame-stacking approach here.)

# Problems With 