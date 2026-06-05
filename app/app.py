import gradio as gr
import numpy as np
import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Label mappings from saved encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
id2label = {i: label for i, label in enumerate(le.classes_)}
label2id = {label: i for i, label in enumerate(le.classes_)}

# Load models
model_name = "./bert_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)
bert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

with open("random_forest_bert_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Inference
def predict(text: str) -> dict:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = bert_model.base_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    pred_id = rf_model.predict(embedding)[0]
    probs = rf_model.predict_proba(embedding)[0]
    return {id2label[1]: float(probs[i]) for i in range(len(id2label))}

# Gradio UI + auto REST API
gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Statement", lines=4,
                      placeholder="Type a statement to classify..."),
    outputs=gr.Label(label="Classification", num_top_classes=7),
    title="Mental Health Sentiment Analysis",
    description="BERT Embeddings + Random Forest Classifier"
).launch()