import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import torch
from torch.utils.data import Dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")


# Function to read the files and create a dataset
def read_files(english_path, german_path):
    with open(english_path, 'r', encoding='utf-8') as f:
        english_lines = f.readlines()
    with open(german_path, 'r', encoding='utf-8') as f:
        german_lines = f.readlines()
    return english_lines, german_lines


# Define paths to the text files
en_path = "de-en-data/en"
de_path = "de-en-data/de"

# Read the files
english_texts, german_texts = read_files(en_path, de_path)

# Create a dataset
dataset = {"de": german_texts, "en": english_texts}


# Preprocess the data
def preprocess_function(examples):
    inputs = [ex for ex in examples["de"]]
    targets = [ex for ex in examples["en"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = preprocess_function(dataset)


# Convert lists to dataset format
class TranslationDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.labels = tokenized_data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }
        return item


train_dataset = TranslationDataset(tokenized_datasets)
eval_dataset = TranslationDataset(tokenized_datasets)  # Adjust as needed

# Load the pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-llama")
tokenizer.save_pretrained("./fine-tuned-llama")

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./fine-tuned-llama")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-llama")

# Create a translation pipeline
translation_pipeline = pipeline("translation_de_to_en", model=model, tokenizer=tokenizer)


# Function to translate German text to English
def translate_article(german_text):
    translated_text = translation_pipeline(german_text)
    return translated_text[0]['translation_text']


# Example usage
german_article = "Es war für Robert Habeck die erste Reise nach Südkorea und China als Minister. Er wollte vor allem reden, Eindrücke sammeln, die europäische Position klarmachen. So ein Blick von außen auf das eigene Land eiche aber auch den Kompass, zeigt sich Habeck am Ende der Reise selbstkritisch."
english_translation = translate_article(german_article)
print(english_translation)
