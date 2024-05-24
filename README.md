# RT-mDL: Supporting Real-Time Mixed Deep Learning Tasks on Edge Platforms

## Introduction

RT-mDL is a novel framework designed to support the execution of mixed real-time deep learning (DL) tasks on edge platforms equipped with heterogeneous CPU and GPU resources. It optimizes the execution of these mixed DL tasks to satisfy diverse real-time and accuracy requirements by leveraging the unique computational characteristics of DL tasks.

## Features

- **Storage-Bounded Model Scaling:** Generates model variants with different workloads and accuracies under user-specified storage constraints.
- **Priority-Based Scheduling:** Employs independent CPU/GPU task queues to enhance CPU/GPU utilization.
- **GPU Packing Mechanism:** Improves GPU spatial utilization by enabling parallel execution of DL inferences with priority guarantees.
- **MOEA-Based Strategy Optimization:** Uses a Multi-Objective Evolutionary Algorithm to find optimal execution strategies that meet real-time requirements and minimize accuracy loss.

## Prerequisites

- Python 3.x
- PyTorch
- CUDA Toolkit
- NVIDIA Jetson TX2, Nano, or AGX Xavier -> to be implemented, it might be used later
- LibTorch (C++ frontend of PyTorch) -> to be implemented, it might be used later
- C++11 Compiler -> to be implemented, it might be used later

