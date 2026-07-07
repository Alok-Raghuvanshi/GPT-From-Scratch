# 🤖 GPT From Scratch using PyTorch

A decoder-only GPT language model implemented completely from scratch using **PyTorch**. This project demonstrates the implementation of the core Transformer architecture without relying on pre-trained language model libraries.

## 🚀 Features

* Decoder-only GPT architecture
* Multi-Head Self-Attention
* Positional Embeddings
* Feed Forward Network
* Residual Connections
* Layer Normalization
* Autoregressive Text Generation
* Streamlit Web Application
* Model Saving and Loading

## 🛠️ Tech Stack

* Python
* PyTorch
* Streamlit

## 📂 Project Structure

```
GPT-From-Scratch/
│── app.py
│── model.py
│── generate.py
│── gpt_model.pth
│── vocab.pkl
│── requirements.txt
└── README.md
```

## 🧠 Model Architecture

* Vocabulary Size: 13,331
* Embedding Dimension: 128
* Context Length: 64
* Transformer Layers: 4
* Attention Heads: 4
* Feed Forward Dimension: 512
* Activation Function: GELU

## 📖 Dataset

The model was trained on the Tiny Shakespeare dataset to generate Shakespeare-style dialogue.

## 💡 Example

**Input**

```
ROMEO :
```

**Output**

```
ROMEO : Ay, thou art a wicked villain, despite of all grace.
First Gentleman : Well, there went but a pair of shears between us.
```

## ▶️ Run the Project

Install the required packages:

```
pip install -r requirements.txt
```

Run the Streamlit application:

```
streamlit run app.py
```

## 📌 Future Improvements

* Top-k and Top-p sampling
* Temperature-controlled generation
* Byte Pair Encoding (BPE) tokenizer
* Model checkpointing
* Hugging Face dataset integration
* Larger GPT architecture

## 👨‍💻 Author

**Alok Raghuvanshi**

B.Tech Information Technology
Harcourt Butler Technical University (HBTU), Kanpur
