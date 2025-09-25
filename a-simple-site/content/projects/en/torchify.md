---
date: 2024-10-19T16:53:53-04:00
# description: ""
# image: ""
lastmod: 2024-10-19
showTableOfContents: false
tags: ["ml",]
title: "Torchify: Compiling a json-like file to a torch.nn.Module"
type: "post"
---
Torchify is a small compiler project that reads a high-level specification of neural network architectures and control logic, then emits equivalent PyTorch code — complete with optimizations like constant folding, dead-code elimination, and common subexpression elimination.

## Motivation & Goal

When defining neural networks or model architectures by hand, boilerplate abounds. Torchify automatically generates PyTorch modules from more declarative specifications. The idea is to enable:

- Faster prototyping: write compact specification DSL instead of full Python classes  
- Maintainability: changes in architecture or control logic are easier to express  
- Optimizations: perform static analysis and simplification before runtime  

## Design & Pipeline

Torchify’s architecture follows a classic compiler pattern:

1. **Lexical & Syntax Analysis**: tokenizing and parsing the DSL specification into a parse tree / AST  
2. **Intermediate Representation & Optimizations**: applying passes like constant folding, copy propagation, algebraic simplification, common subexpression elimination, and dead-code elimination
3. **Code Generation**: translating the optimized AST into a Python `nn.Module` subclass, wiring layers and forward logic accordingly

Each optimization pass is repeated until reaching a fixed point (i.e. no further changes). The generated code aims to be clean and readable, preserving expressiveness rather than over-optimizing every constant expression.

## Usage Example

Suppose you write a DSL like:

```text
module {
  linear0: {
    dim_in = 784;
    dim_out = 128;
  }
  linear1: {
    dim_in = dim_out of linear0;
    dim_out = 10;
  }
  if (some_condition) {
    // optional branch
    linear2: { … }
  }
}
