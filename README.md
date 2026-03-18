# 🤖 LLM Fine-Tuning Lab: GPT-2

A research-oriented repository focused on fine-tuning Large Language Models (LLMs), specifically **GPT-2**, using various optimization techniques. 

This project explores the trade-offs between **Full Parameter Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA**.

---

## 🎯 Project Goals

* **Compare Performance:** Evaluation of LoRA vs. standard full-parameter fine-tuning.
* **Hyperparameter Analysis:** Analyzing the impact of learning rates, epochs, and LoRA rank ($r$).
* **Model Adaptability:** Testing weights across diverse datasets to observe stylistic and functional shifts.

---

## 🛠 Tech Stack & Libraries

* **Model:** GPT-2 (Base/Medium)
* **Framework:** [PyTorch](https://pytorch.org/) & [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* **Optimization:** `peft` (LoRA)

---

## 🔬 Methodology

### 1. Full Parameter Fine-Tuning
Updating all weights in the GPT-2 architecture. This serves as the baseline for performance but requires significant VRAM.

### 2. LoRA (Low-Rank Adaptation)
Injecting trainable rank decomposition matrices into the transformer layers while freezing the original weights.
* **Benefit:** Drastically reduces the number of trainable parameters.
* **Efficiency:** Lower memory footprint, making it possible to train on consumer GPUs.

