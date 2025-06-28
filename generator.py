import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_SDP_BACKEND"] = "math"

class Generator:
    def __init__(self, name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tok   = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
            attn_implementation="eager"
        ).eval()

    def generate(self, question, ctx):
        prompt = "\n".join(ctx) + "\n\n" + question + "\n정답:"
        ids    = self.tok(prompt, return_tensors="pt").input_ids
        out    = self.model.generate(
            ids,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tok.eos_token_id
        )
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()