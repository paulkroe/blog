---
date: 2024-08-01T16:53:53-04:00
# description: ""
# image: ""
lastmod: 2024-08-01
showTableOfContents: false
tags: ["rl",]
title: "Schokoban: Monte Carlo Tree Search for Solving Sokoban"
type: "post"
---
Schokoban is a Sokoban solver, which unlike state-of-the-art Sokoban solvers that often use heuristic or search-based algorithms, experiments with applying **Monte Carlo Tree Search (MCTS)** to this classic puzzle game.

The project explores how to adapt MCTS to handle the **unique challenges of Sokoban**, such as redundant states and branching complexity. A key innovation is a new method for managing redundant states within the tree search, resulting in substantial performance improvements over a baseline approach.

While the solver’s main goal is *experimental research* rather than setting new performance records, it can still successfully solve most puzzles in well-known level collections like [Microban III](http://www.abelmartin.com/rj/sokobanJS/Skinner/David%20W.%20Skinner%20-%20Sokoban.htm).

If you want to learn more, see performance results, feel free to explore the code on GitHub:
[https://github.com/paulkroe/Schokoban](https://github.com/paulkroe/Schokoban)

![Example Sokoban level](/images/schokoban.png)
*Example Sokoban level from by [Skinner](http://www.abelmartin.com/rj/sokobanJS/Skinner/David%20W.%20Skinner%20-%20Sokoban.htm).*