
import gradio as gr
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

login(token=os.environ.get("HF_TOKEN"))

model_id = "harsha85018/llama-medical-summarizer"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

def summarize(article):
    input_text = f"### Article:\n{article[:512]}\n\n### Summary:\n"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated.split("### Summary:")[-1].strip()

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, placeholder="Paste a medical article here...", label="Medical Article"),
    outputs=gr.Textbox(lines=5, label="Generated Summary"),
    title="Medical Text Summarizer",
    description="Fine-tuned LLaMA 3.2 (1B) with LoRA/QLoRA on 10K+ PubMed abstracts."
)

demo.launch()
