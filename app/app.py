import json
import os
import gradio as gr
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "./distilbert_onnx_int8"
MAX_LEN = 128

# Locate the quantized graph
onnx_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".onnx")]
if not onnx_files:
    raise FileNotFoundError(f"No .onnx file found in {MODEL_DIR}")
onnx_name = "model_quantized.onnx" if "model_quantized.onnx" in onnx_files else onnx_files[0]
onnx_path = os.path.join(MODEL_DIR, onnx_name)

# Labels from the model config
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    config = json.load(f)
id2label = {int(k): v for k, v in config["id2label"].items()}
LABELS = [id2label[i] for i in range(len(id2label))]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = max(1, os.cpu_count() or 1)
session = ort.InferenceSession(
    onnx_path, session_options, providers=["CPUExecutionProvider"]
)
INPUT_NAMES = {i.name for i in session.get_inputs()}

print(f"Loaded {onnx_name} | labels: {LABELS}")

HIGH_RISK_LABELS = {"Suicidal"}
RISK_THRESHOLD = 0.40


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def predict(text: str) -> dict:
    if not text or not text.strip():
        return {label: 0.0 for label in LABELS}

    encoded = tokenizer(
        text, return_tensors="np", truncation=True, padding=True, max_length=MAX_LEN
    )

    feed = {k: v.astype(np.int64) for k, v in encoded.items() if k in INPUT_NAMES}

    logits = session.run(None, feed)[0]
    probs = _softmax(logits)[0]

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


DESCRIPTION = (
    "Research prototype, **not a diagnostic tool.** This model screens text for "
    "possible indicators of psychological distress and is not a substitute for "
    "assessment by a licensed clinician."
)

ARTICLE = (
    "### If you or someone you know needs support\n"
    "- **US:** call or text **988** (Suicide & Crisis Lifeline), or text **HOME** to **741741**\n"
    "- **UK & ROI:** call **116 123** (Samaritans)\n"
    "- **International:** [findahelpline.com](https://findahelpline.com)\n\n"
    "This tool is a research prototype trained on public social-media text. Its "
    "labels are derived from posting venue rather than clinical assessment, and it "
    "can make mistakes."
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Statement", lines=4, placeholder="Type a statement to classify..."
    ),
    outputs=gr.Label(label="Screening result", num_top_classes=7),
    title="Mental Health Text Screening (Research Prototype)",
    description=DESCRIPTION,
    article=ARTICLE,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()