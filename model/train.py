# %%
import pandas as pd
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration

# %% [markdown]
# Data collection
# 

# %% [markdown]
# 

# %%
train_data = pd.read_csv("train_dataset.csv")
val_data =  pd.read_csv("validation_dataset.csv")

# %%
train_data.shape

# %%
val_data.head()

# %%
val_data.shape

# %%
# random sampling
train_dataset = train_data.sample(n=14000, random_state=42).reset_index(drop=True)
val_dataset = val_data.sample(n=500, random_state=42).reset_index(drop=True)

# %%
train_dataset.shape

# %%


# %% [markdown]
# Data preprocessing
# 

# %%
import re

def clean_data(text):
  text = str(text)
  text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
  text = re.sub(r"\r\n", " ", text) #Lines
  text = re.sub(r"\s+", " ", text) #extra spaces
  text = re.sub(r"<.*?>", " ", text) #html tags
  text = text.strip().lower()

  return text


# %%
# print(train_dataset["dialogue"].isna().sum())
# print(train_dataset["summary"].isna().sum())

# %%
train_dataset["dialogue"] = train_dataset["dialogue"].apply(clean_data)
train_dataset["summary"] = train_dataset["summary"].apply(clean_data)

val_dataset["dialogue"] = val_dataset["dialogue"].apply(clean_data)
val_dataset["summary"] = val_dataset["summary"].apply(clean_data)

# %%
train_dataset["dialogue"][0]

# %% [markdown]
# Tokenization

# %%
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# %%
def tokenize(data):
    dialogue = str(data['dialogue'])
    summary = str(data['summary'])

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True
    )

    targets = tokenizer(
        summary,
        padding="max_length",
        max_length=150,
        truncation=True
    )

    inputs["labels"] = targets["input_ids"]

    return inputs

# %%
train_dataset = train_data.apply(tokenize, axis=1).tolist()
val_dataset = val_data.apply(tokenize, axis=1).tolist()

# %%
type(train_dataset)

# %%
model = T5ForConditionalGeneration.from_pretrained("t5-small")



# %% [markdown]
# Working with the model
# 

# %%
import torch

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
except ImportError:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

print("The device:", device)

# %%
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=100,
    optim="adamw_torch"
)

# %%
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset
)

# %%
# to train the model

trainer.train()

# %%
import torch_xla.core.xla_model as xm

xm.mark_step()
model = model.to("cpu")

model.save_pretrained("./saved_summary_model", safe_serialization=False)
tokenizer.save_pretrained("./saved_summary_model")

# %%
import shutil
from google.colab import files

# Zip the folder
shutil.make_archive("saved_summary_model", "zip", "./saved_summary_model")

# Download it
files.download("saved_summary_model.zip")

# %%
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# %% [markdown]
# # New section

# %% [markdown]
# Test the core logic for summarization
# 

# %%
def summarize_dialogue(dialogue):
  dialogue = clean_data(dialogue) #clean

  #tokenize
  inputs = tokenizer(
      dialogue,
      padding="max_length",
      max_length=512,
      truncation=True,
      return_tensors="pt"
  ).to(device)

  #generate the summary based on id
  model.to(device)
  targets = model.generate(
      inputs_ids = inputs["inputs_ids"],
      attention_mask = inputs["attention_mask"],
      max_length = 150,
      num_beams = 4,
      early_stopping = True
  )

  summary = tokenizer.decode(targets[0],skip_special_tokens=True) #EOS, SEP
  return summary


# %%
# test_dailogue =
# """
# data from dataset

# """
# summary  = summarize_dialogue(test_dailogue)

# print("Summary: ", summary)


