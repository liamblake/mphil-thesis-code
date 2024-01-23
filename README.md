This repository contains all the Julia code that was used to generate the results and figures presented in my MPhil thesis, titled "Computable Characterisations of Uncertainty in Differential Equations" and completed at the University of Adelaide in early 2023.
The thesis itself is available [here](https://github.com/liamblake/mphil-thesis).

## Contents

- `computations`: implementations of the Gaussian limit, stochastic sensitivity, and Gaussian mixture model computations, which are used across the remaining subdirectories.

- `sde-linearisation-bounds`: the standalone repository used to generate all figures for Chapter 3 ('Characterising SDE linearisations: the numerics'). These figures were also used in the submitted article *The convergence of stochastic differential equations to their linearisation in small noise limits*, available as a preprint on arXiv [here](https://arxiv.org/abs/2309.16334).

- `benes-sde`: the demonstrative implementation of the Gaussian mixture model on Benê's SDE in Chapter 4 ('A Gaussian Mixture Model').

- `gulf_stream`:

- `epidemiology`: the CTMC simulations and Gaussian computation for the discussion on population processes in the final chapter (Section 7.7).
