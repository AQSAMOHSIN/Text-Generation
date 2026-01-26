import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class GPT2fineTuned:
    def __init__(self, weights_path="Text-Generation/model.safetensors"):
        self.weights_path = weights_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        state_dict = load_file(self.weights_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    def generate_text(self, prompt, max_new_tokens=120, temperature=0.8, top_p=0.9, top_k=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
