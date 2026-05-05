from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from collections import defaultdict

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


def parse_conversation(text: str):
    """
    Parses lines like:
        Amanda: I baked cookies. Do you want some?
        Jerry: Sure!
    Returns:
        speakers_order : list of speaker names in order of appearance
        speaker_lines  : dict mapping speaker -> list of their lines
        all_lines      : list of (speaker, text) tuples in original order
    """
    pattern = re.compile(r"^([A-Za-z][A-Za-z0-9 _\-]*):\s*(.+)$")
    speaker_lines = defaultdict(list)
    speakers_order = []
    all_lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            speaker = match.group(1).strip()
            content = match.group(2).strip()
            if speaker not in speakers_order:
                speakers_order.append(speaker)
            speaker_lines[speaker].append(content)
            all_lines.append((speaker, content))

    return speakers_order, dict(speaker_lines), all_lines


def first_to_third_person(text: str, name: str) -> str:
    """
    Converts first-person pronouns in a sentence to third-person using
    the speaker's name.

    WHY: When T5 receives "I'll bring cookies tomorrow", it doesn't know
    who "I" is and mixes up speakers. Replacing with the actual name
    ("Amanda will bring cookies tomorrow") keeps each person's summary
    accurate and distinct.

    Handles common patterns:
        I am / I'm  → Name is
        I have      → Name has
        I will / I'll → Name will
        I           → Name
        my          → Name's
        me          → Name
        myself      → Name
        we / our    → kept as-is (shared actions)
    """
    # Work sentence by sentence to avoid partial replacements
    replacements = [
        (r"\bI'm\b",    f"{name} is"),
        (r"\bI've\b",   f"{name} has"),
        (r"\bI'll\b",   f"{name} will"),
        (r"\bI'd\b",    f"{name} would"),
        (r"\bI\b",      name),
        (r"\bmy\b",     f"{name}'s"),
        (r"\bme\b",     name),
        (r"\bmyself\b", name),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def build_speaker_summary_input(speaker: str, speaker_lines: dict) -> str:
    """
    Converts a speaker's lines to third-person and joins them into a
    single paragraph for T5 to summarize.

    This ensures T5 only sees what THIS speaker said, with clear
    attribution (name instead of "I"), preventing cross-speaker confusion.
    """
    converted = []
    for line in speaker_lines[speaker]:
        converted.append(first_to_third_person(line, speaker))
    return " ".join(converted)


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

    # Check if this is a multi-person conversation
    speakers_order, speaker_lines, all_lines = parse_conversation(dialogue)

    if len(speakers_order) >= 2:
        # Step 1: Per-person summaries using 3rd-person converted input
        # Each speaker's lines are converted from "I/my/me" → "Name/Name's/Name"
        # so T5 knows exactly who owns each action.
        per_person = []
        for speaker in speakers_order:
            third_person_text = build_speaker_summary_input(speaker, speaker_lines)

            if len(third_person_text.split()) >= 10:
                person_summary = summarize_dialogue(third_person_text)
            else:
                # Too short to summarize — use converted text directly
                person_summary = third_person_text

            per_person.append({
                "name": speaker,
                "summary": person_summary
            })

        # Step 2: Overall summary built from per-person summaries
        # Guarantees every speaker's name appears in the overall summary
        combined_for_overall = " ".join(
            f"{p['name']}: {p['summary']}" for p in per_person
        )
        overall_summary = summarize_dialogue(combined_for_overall)

        return jsonify({
            "summary": overall_summary,
            "per_person": per_person,
            "is_conversation": True
        })
    else:
        # Plain text or single speaker
        summary = summarize_dialogue(dialogue)
        return jsonify({
            "summary": summary,
            "is_conversation": False
        })


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)