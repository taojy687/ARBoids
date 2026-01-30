# ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense

## 📄 Paper
This repository contains the code implementation of our [paper](https://arxiv.org/abs/2502.18549), which has been accepted by **IEEE Robotics and Automation Letters (RA-L)**.

## 🔍 Overview
We focus on the target defense problem (TDP) for multiple unmanned surface vehicles (USVs), which concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, we introduce ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. A novel adapter module is designed to dynamically balance the weights between base policy and RL policy.

🚧 We are currently organizing and cleaning up the source code. It will be uploaded to this repository as soon as it is ready.
