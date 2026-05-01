from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

app = Flask(__name__)

# Model loading
model = T5ForConditionalGeneration.from_pretrained(
    "./model/saved_summary_model",
    local_files_only=True
)
tokenizer = T5Tokenizer.from_pretrained(
    "./model/saved_summary_model",
    local_files_only=True
)

# Device setup 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()  

# Text cleaning 
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()
    # FIX 2: removed .lower() — preserves proper nouns & acronyms like RAG, T5
    return text

# Summarization
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad(): 
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=130,   
            min_length=30,    
            num_beams=8,
            length_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# Routes 
@app.route("/summarize/", methods=["POST"])
def summarize():
    data = request.get_json()

    if not data or "dialogue" not in data:
        return jsonify({"error": "Missing 'dialogue' field"}), 400

    dialogue = data["dialogue"].strip()

    if not dialogue:                        
        return jsonify({"error": "Dialogue cannot be empty"}), 400

    if len(dialogue) > 5000:                
        return jsonify({"error": "Input too long. Max 5000 characters."}), 400

    summary = summarize_dialogue(dialogue)
    return jsonify({"summary": summary})


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)