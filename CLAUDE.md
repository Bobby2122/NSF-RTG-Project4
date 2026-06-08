# Project Notes for Claude

## Presentation Notes (to incorporate once convergence data is complete)

### sin(2πx) — Best Case for the Conjecture
sin(2πx) with k=3 is the clearest example of the conjecture working cleanly. Its three
evenly spaced inflection points (x = -1/2, 0, +1/2) create strong, balanced gradient
attractors that neurons organize around from the very start — even at small m. The cluster
count C stays near k=3 throughout the m sweep with no rise-then-fall artifact.

This is worth highlighting in the presentation as evidence that **inflection point
structure, not just count, shapes convergence behavior**. The gradient field geometry
of f* determines how easily and how early the network finds the k cluster positions.

Contrast with:
- sin(πx) k=1: one weak attractor, neurons form intermediate clusters before collapsing
- sin(4πx) k=7: many clusters needed, small m can't populate all positions so C < k first
- x³ k=1: polynomial target, very slow convergence — needs much larger m

### General Presentation Status
- Waiting on simulate_parallel.py Phase 1 + Phase 2 to complete before finalizing plots
- presentation/presentation.tex already exists with initial slides
- Once new convergence data is in, update convergence_plot_parallel.png and recompile
