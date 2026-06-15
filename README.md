# Fine-Tuned LLaMA for Medical Text Summarization

Fine-tuned **LLaMA 3.2 (1B)** with **LoRA/QLoRA** on 10,000+ PubMed abstracts for medical text summarization.

## Tech Stack
`Python` `PyTorch` `HuggingFace Transformers` `PEFT` `LoRA` `QLoRA` `Gradio` `bitsandbytes`

## Results
- **ROUGE-L Score:** 0.1821 (3 epochs, 10K samples)
- **Trainable Parameters:** 1.7M / 1.2B (0.14% via LoRA)
- **Training Time:** ~3.5 hours on T4 GPU

## Demo
[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/harsha85018/medical-text-summarizer)

## Model
[HuggingFace Model Hub](https://huggingface.co/harsha85018/llama-medical-summarizer)

## How It Works
1. Base model: LLaMA 3.2-1B loaded in 4-bit quantization (QLoRA)
2. LoRA adapters applied to q_proj and v_proj layers (r=16, alpha=32)
3. Fine-tuned on PubMed article-abstract pairs
4. Gradio interface for real-time summarization

## Run Locally
pip install transformers peft bitsandbytes accelerate gradio torch
python app.py
