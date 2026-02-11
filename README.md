Multi-Sensor Fusion for Autonomous Robot Navigation

Bayesian Pattern Recognition and Conditional Generative Mapping
Rikkin Mehta — Texas A&M University

1. Project Overview

In this project, I developed a complete perception-to-navigation pipeline for autonomous robots that operate under uncertainty and sensor limitations. Real-world robotic systems cannot depend on a single sensing modality:

RGB cameras degrade in poor lighting and motion blur

Depth sensors struggle with reflective or distant surfaces

LiDAR provides geometry but lacks semantic richness

To address these limitations, I designed a system that fuses RGB, Depth, and 360-degree LiDAR using Bayesian multi-sensor fusion and integrates conditional generative modeling to support safe navigation. This project combines pattern recognition, probabilistic decision theory, classical machine learning, deep learning, and robotics planning into a unified framework.

2. System Architecture

I structured the system as a full robotics perception-to-planning pipeline:

Sensor data acquisition (RGB, Depth, LiDAR)

Modality-specific feature extraction

Per-modality classification

Bayesian late fusion of posterior probabilities

Conditional GAN-based semantic cost map generation

Graph-based path planning using A* or Dijkstra


End-to-end perception-to-navigation architecture used in this project.

3. Dataset and Simulation Environment

To evaluate the system, I generated a high-fidelity synthetic dataset in Webots using physics-based sensor models in a warehouse-like environment with static obstacles, moving objects, and goal regions.

Each data sample includes:

Sensor	Data Description
RGB Camera	640 × 480 image
Depth Sensor	0–10 m depth map
LiDAR	360-degree scan
Label	Free / Static Obstacle / Dynamic Obstacle / Goal

The dataset contains 1,000 synchronized multi-modal samples, split into:

70% training

15% validation

15% testing


Examples of multi-sensor observations used for training and evaluation.

4. Feature Engineering

I designed modality-specific feature pipelines to capture complementary information while controlling dimensionality.

RGB Features (736 dimensions)

HOG descriptors for edges and textures

Color histograms

ResNet-18 embeddings for deep semantic representation

Depth Features (262 dimensions)

Surface normals for geometric structure

Occupancy encodings

Distance distribution statistics

LiDAR Features (53 dimensions)

360-degree range histograms

DBSCAN cluster summaries

Range distribution moments


Feature dimensionality comparison across modalities.

5. Classification Framework

For each modality, I trained multiple classifiers:

Classical machine learning models:

Naive Bayes

Support Vector Machine (RBF kernel)

Random Forest

Deep learning model:

Multi-Layer Perceptron (MLP) trained on engineered features

Each model produces posterior class probabilities, which are later used for fusion.

6. Bayesian Multi-Sensor Fusion

I implemented a Bayesian late fusion strategy that combines posterior probabilities from each modality:

P(y | x) = w_RGB * P(y | x_RGB)
+ w_Depth * P(y | x_Depth)
+ w_LiDAR * P(y | x_LiDAR)

where the weights satisfy:

w_RGB + w_Depth + w_LiDAR = 1

These weights reflect sensor reliability. Under sensor degradation, the fusion mechanism can shift trust toward more reliable modalities.

7. Conditional GAN for Semantic Cost Maps

Beyond classification, I trained a Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate dense semantic cost maps conditioned on fused sensor features.

Generator objective:
LG = - E[D(C_hat, c)]

Critic objective:
LD = E[D(C, c)] - E[D(C_hat, c)] + lambda * GradientPenalty

The cost map represents spatial traversal cost:

High values indicate obstacles

Low values indicate free space


Generated semantic cost map and collision-free navigation path.

8. Planning Module

I integrated the generated cost maps with graph search algorithms:

Dijkstra

A*

to compute minimum-cost collision-free trajectories from start to goal, completing the perception-to-planning loop.

9. Experimental Evaluation

I evaluated the system using five-fold stratified cross-validation.

All models achieved near-perfect performance due to strong class separability in the synthetic dataset, resulting in a ceiling effect. Fusion benefits are expected to become more visible under realistic sensor noise and domain shifts.

10. Limitations

The dataset is synthetic and highly separable

Fusion advantages are masked without sensor degradation

Real-world deployment remains future work

11. Future Work

In future extensions, I plan to include:

Sensor degradation experiments (noise, blur, dropout)

Domain shift testing

Correlation-aware fusion methods

Hardware-in-the-loop deployment

12. Technology Stack

Python

PyTorch (GAN implementation)

Scikit-learn (classifiers)

NumPy and OpenCV

Webots simulation



13. Key Insight

Through this project, I demonstrate how probabilistic multi-sensor fusion combined with generative semantic mapping can form a complete perception-to-navigation stack. This approach bridges classical pattern recognition and modern deep learning, providing a robust framework for autonomous robotics perception and navigation.
