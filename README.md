# Flowing Through States: Neural ODE Regularization for Reinforcement Learning

## Project Overview
This repository contains a reproducibility study of the paper **["Flowing Through States: Neural ODE Regularization for Reinforcement Learning"](https://openreview.net/forum?id=FHFDCsB9UC)**.

 The project is developed as the practical examination for the **Advanced Topics in Machine Learning** course, part of my Master’s Degree in Artificial Intelligence (AY 2025-2026).

## The Problem: Latent Misalignment
In deep RL, agents typically rely on latent representations of environment states to make decisions. While the environment's semantic dynamics dictate how states evolve, the corresponding transitions in the latent space are often left implicit and unconstrained. This can lead to a misalignment between the agent's internal representation and the actual structure of the Markov Decision Process (MDP).

## The Approach & Study Purpose
To tackle this misalignment, the authors introduce **Flow Regularization (FlowReg)**. This unsupervised technique explicitly models latent dynamics by training a Neural Ordinary Differential Equation (ODE) to act as a continuous surrogate for the environment. By applying an alignment penalty, the agent's latent embeddings are forced to mimic the smooth flows of the Neural ODE, inheriting its topological consistency.

The purpose of this project is to build and replicate the FlowReg framework from scratch. We will integrate this technique on top of established algorithms (like A2C for Atari and PPO for Gridworld) to critically validate the performance gains and latent smoothness properties reported in the original study.

For a deep dive into the mathematical proofs and complete theoretical framework, please refer to the original paper: [link](https://openreview.net/forum?id=FHFDCsB9UC).