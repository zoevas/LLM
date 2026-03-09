# Medical AI Assistant (Phi-2 Fine-Tuning)

This project demonstrates how to fine-tune a language model to create a **domain-specific medical question answering assistant** using parameter-efficient training techniques.

The base model used is **Microsoft Phi-2**, which was fine-tuned using **LoRA (Low-Rank Adaptation)** on a medical Q&A dataset.  
The goal of this project is to explore how small language models can be specialized for specific domains using limited datasets and consumer hardware.

---

## Project Overview

Large Language Models (LLMs) can be adapted to specialized domains through **fine-tuning**. In this project, a small medical assistant was created that can answer basic health-related questions.

Key ideas explored in this project:

- Fine-tuning a pre-trained language model
- Parameter-efficient training with LoRA
- Running LLM experiments on consumer GPUs
- Instruction-style prompt formatting for training

---

## Base Model

The model used for fine-tuning:

**Microsoft Phi-2**

https://huggingface.co/microsoft/phi-2

Phi-2 is a lightweight language model that performs well for research experiments and local machine learning projects.

---

## Dataset

The dataset used for training:

**Simple Medical Q&A Dataset (JSONL)**  
https://www.kaggle.com/datasets/mouadenna/simple-q-and-a-medical-dataset-jsonl

The dataset contains medical question-answer pairs stored in JSONL format.

Example entry from the dataset:

```json
{
  "question": "what is malaria?",
  "answer": "Malaria is an infectious disease caused by Plasmodium parasites transmitted by infected mosquitoes."
}
```

## Hardware

Training was performed on:
GPU: NVIDIA GeForce RTX 3060

This project demonstrates that fine-tuning language models is possible on consumer GPUs using efficient training methods like LoRA.

## How install it
```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python fine_tune.py
```