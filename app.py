import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import pickle

from model import GPTModel
from generate import generate

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="GPT From Scratch",
    page_icon="🤖",
    layout="wide"
)

# -------------------- Load Vocabulary --------------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# If vocab.pkl stores a dictionary
if isinstance(vocab, dict):
    word_to_idx = vocab
    idx_to_word = {v: k for k, v in vocab.items()}

# If vocab.pkl stores a list
elif isinstance(vocab, list):
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

else:
    st.error("Unsupported vocabulary format.")
    st.stop()

device = torch.device("cpu")


# -------------------- Load Model --------------------
@st.cache_resource
def load_model():

    model = GPTModel(
        vocab_size=len(word_to_idx),
        embedding_dim=128,
        block_size=64,
        num_heads=4,
        num_layers=4
    ).to(device)

    model.load_state_dict(
        torch.load(
            "gpt_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model


model = load_model()


# -------------------- UI --------------------
st.title("🤖 GPT From Scratch")
st.caption("Decoder-only Transformer implemented from scratch using PyTorch")

with st.sidebar:

    st.header("Generation Settings")

    max_tokens = st.slider(
        "Max New Tokens",
        min_value=10,
        max_value=100,
        value=50
    )

prompt = st.text_area(
    "Enter Prompt",
    value="ROMEO :",
    height=120
)


# -------------------- Generate --------------------
if st.button("Generate Text"):

    with st.spinner("Generating..."):

        text = generate(
            model=model,
            start_text=prompt,
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            device=device,
            block_size=64,
            max_new_tokens=max_tokens
        )

    st.success("Generation Complete!")

    st.code(text, language="text")