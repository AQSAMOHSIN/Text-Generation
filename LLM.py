from os import path
import torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math
from n_gram import BackoffNGramLanguageModel
import random


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

    def eval_bpc_and_ppl(self, texts, stride=512):
        """
        Fair metric vs char-RNN/LSTM: BPC (bits per character)

        texts: list[str] or str
        stride: sliding window step (handles long texts)
        """
        if isinstance(texts, str):
            texts = [texts]

        # IMPORTANT: use the SAME preprocessing as your char models
        # (e.g., lowercase? keep \n? remove extra spaces?) to be fair.
        full_text = "\n".join(texts)
        total_chars = len(full_text)
        if total_chars == 0:
            raise ValueError("Empty evaluation text.")

        enc = self.tokenizer(full_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        max_len = getattr(self.model.config, "n_positions", 1024)
        seq_len = input_ids.size(1)

        total_nll_nats = 0.0
        total_scored_tokens = 0

        prev_end = 0
        for start in range(0, seq_len, stride):
            end = min(start + max_len, seq_len)
            trg_len = end - prev_end  # score only new tokens introduced this step

            input_chunk = input_ids[:, start:end]
            labels = input_chunk.clone()

            # Mask overlap (don’t rescore context)
            if trg_len < labels.size(1):
                labels[:, :-trg_len] = -100

            outputs = self.model(input_chunk, labels=labels)
            # outputs.loss is mean over non-masked positions (in nats)
            total_nll_nats += outputs.loss.item() * trg_len
            total_scored_tokens += trg_len

            prev_end = end
            if end == seq_len:
                break

        # Token-level perplexity (not comparable to char ppl directly)
        avg_nll_nats_per_token = total_nll_nats / max(total_scored_tokens, 1)
        token_ppl = math.exp(avg_nll_nats_per_token)

        # BPC: (total NLL in bits) / (#chars)
        total_nll_bits = total_nll_nats / math.log(2)
        bpc = total_nll_bits / total_chars

        return {
            "bpc": bpc,
            "token_ppl": token_ppl,
            "total_chars": total_chars,
            "scored_tokens": total_scored_tokens,
        }


gpt = GPT2fineTuned("Text-Generation/model.safetensors")
path = "/Users/laibaqureshi/Desktop/Text Generation/Text-Generation/shakespeare_2.txt"
sentences = BackoffNGramLanguageModel(path, n=3).load()
random.shuffle(sentences)  # important: avoid “last 20% is different”

split = int(0.8 * len(sentences))
train_sents = sentences[:split]
test_sents = sentences[split:]
metrics = gpt.eval_bpc_and_ppl(test_sents)

print("GPT-2 BPC:", metrics["bpc"])
print("GPT-2 token perplexity (FYI):", metrics["token_ppl"])
