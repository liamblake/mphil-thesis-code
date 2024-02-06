This repository contains all the Julia code that was used to generate the results and figures presented in my MPhil thesis, titled "Computable Characterisations of Uncertainty in Differential Equations" and completed at the University of Adelaide in early 2023.
The thesis itself is available [here](https://github.com/liamblake/mphil-thesis).

## Contents

- `computations`: implementations of the Gaussian limit, stochastic sensitivity, and Gaussian mixture model computations, which are used across the remaining subdirectories.

- `background`: the code used to generate the figures in Chapter 2 ('Background') that show realisations of the Wiener process and solutions to a stochastic differential equation.

- `sde-linearisation-bounds`: the standalone repository used to generate figures for Chapter 4 ('Characterising SDE linearisations: the numerics'). These figures were also used in the submitted article *The convergence of stochastic differential equations to their linearisation in small noise limits*, available as a preprint on arXiv [here](https://arxiv.org/abs/2309.16334).

- `gabc`: the computation of the stochastic sensitivity field for the Gromeka-Arnold-Beltrami-Childress flow, to produce figure in Section 4.3.2.

- `benes-sde`: the demonstrative implementation of the Gaussian mixture model on Benê's SDE in Chapter 5 ('A Gaussian mixture model').

- `gulf_stream`: computation of stochastic sensitivity and the Gaussian approximation for an altimetry-driven model of a drifter in the Gulf Stream of the North Atlantic Ocean. This also includes an investigation using two simple implementations of the mixture model. All figures are generated to appear in Chapter 6 ('An application: drifter in the Gulf Stream').

- `epidemiology`: the CTMC simulations and Gaussian computation for the discussion on population processes in the final chapter (Section 7.7).
