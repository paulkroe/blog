---
title: "Schokoban: Using Monte Carlo Tree Search for Solving Sokoban"
date: 2024-07-18
author: "Paul Kroeger"
description: "Bachelor's thesis applying Monte Carlo Tree Search to Sokoban"
toc: false
tags:
---

![Sokoban Puzzle](/level_1.png "Sokoban Puzzle")

*Level 1 from the original Sokoban game*

In my Bachelor's thesis, under the supervision of [Professor Dr. Peters](https://people.math.ethz.ch/~jopeters/), I explored the application of Monte Carlo Tree Search to the puzzle game Sokoban. Sokoban challenges players to push boxes onto designated targets within a warehouse. Even though the rules of Sokoban are simple, they still allow for complex and engaging puzzles. The research presents a novel approach to manage redundant state representations within the Monte Carlo Search Tree, significantly enhancing the algorithm’s ability to solve problems. Below is a preview of the comparative results between the baseline and the improved algorithm.

**Performance Comparison on CBC Level Collection:**

| Number of Iterations | 25 | 50 | 100 | 500 | 1000 | 2000 | 5000 | 10000 | 100000 |
|----------------------|----|----|-----|-----|------|------|------|-------|--------|
| Vanillaban           | 4  | 12 | 16  | 38  | 41   | 44   | 49   | 49    | 50     |
| Schokoban            | 6  | 20 | 41  | 60  | 60   | 60   | 60   | 60    | 60     |

The table shows the number of levels solved out of 60 in the [CBC Level Collection](https://www.cbc.ca/kids/games/play/sokoban).

**Performance Comparison on Microban Level Collection:**

| Number of Iterations | 25 | 50 | 100 | 500 | 1000 | 2000 | 5000 | 10000 | 100000 |
|----------------------|----|----|-----|-----|------|------|------|-------|--------|
| Vanillaban           | 0  | 1  | 7   | 18  | 22   | 29   | 34   | 36    | 44     |
| Schokoban            | 0  | 7  | 18  | 50  | 60   | 70   | 77   | 80    | 90     |

The table shows the number of levels solved out of 100 in the [Microban Level Collection](http://www.abelmartin.com/rj/sokobanJS/Skinner/David%20W.%20Skinner%20-%20Sokoban.htm) by Skinner.

The code developed for this thesis is available on my [GitHub](https://github.com/paulkroe/Schokoban), and the complete thesis can be downloaded [here](/BSc_Thesis_at_SfS_Kroeger.pdf).
