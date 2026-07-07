import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import math
import re

@torch.no_grad()
def generate(
    model,
    start_text,
    word_to_idx,
    idx_to_word,
    device,
    block_size,
    max_new_tokens=50
):

    tokens = re.findall(r"\w+|[^\w\s]", start_text)

    encoded = []
    unknown = []

    for token in tokens:
        if token in word_to_idx:
            encoded.append(word_to_idx[token])
        else:
            unknown.append(token)

    print("Known tokens:", [t for t in tokens if t in word_to_idx])
    print("Unknown tokens:", unknown)

    if len(encoded) == 0:
        return "No known words in vocabulary."

    input_ids = torch.tensor([encoded], device=device)

    for _ in range(max_new_tokens):

        input_crop = input_ids[:, -block_size:]

        logits = model(input_crop)
        logits = logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    words = [idx_to_word[idx.item()] for idx in input_ids[0]]
    return " ".join(words)