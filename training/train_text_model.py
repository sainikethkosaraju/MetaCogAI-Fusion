# train_text_model.py
# Fine-tunes the text expert (DistilBERT) on a 5,000-review subset of IMDb.
# Run from the repository root:  python training/train_text_model.py

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch

# 1. Load IMDb dataset
dataset = load_dataset("imdb", split={'train': 'train', 'test': 'test'})


# 2. Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 3. Tokenize the data
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding=True)

tokenized_datasets = dataset.map(tokenize_fn, batched=True)

# 4. Format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5. Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints/text_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    save_steps=500  # 👈 Just a fallback if save_strategy is not allowed
)



# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(5000)),  # small subset
    eval_dataset=tokenized_datasets["test"].select(range(1000)),
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# 8. Train and save
trainer.train()
trainer.save_model("./weights/text_model")
