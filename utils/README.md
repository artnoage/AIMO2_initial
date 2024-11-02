---
language: en
license: mit
pretty_name: Numina-Olympiads
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
task_ids:
  - math-word-problems
---

# Numina-Olympiads

Filtered NuminaMath-CoT dataset containing only olympiads problems with valid answers.

## Dataset Information
- Split: train
- Original size: 859494
- Filtered size: 129942
- Source: olympiads
- All examples contain valid boxed answers

## Dataset Description
This dataset is a filtered version of the NuminaMath-CoT dataset, containing only problems from olympiad sources that have valid boxed answers. Each example includes:
- A mathematical word problem
- A detailed solution with step-by-step reasoning
- A boxed final answer in LaTeX format

## Usage
The dataset is particularly useful for:
- Training and evaluating math problem-solving models
- Studying olympiad-style mathematical reasoning
- Testing model capabilities on complex word problems
