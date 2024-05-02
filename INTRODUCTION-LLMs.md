# Training a Large Language Model



## Introduction to Large Language Models

If you are unfamiliar with the technical details of large language models (LLMs), I recommend reading (some of) the following resources:

- [A Playlist on DeepLearning from the (objectively) best Math-Channel on YouTube](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) which includes a video on GPTs.
- [The Huggingface Introduction to Transformers](https://huggingface.co/learn/nlp-course/chapter1/1) which is interesting for both the theoretical, as well as practical side of LLMs.
- [A Google Course on LLMs](https://www.cloudskillsboost.google/course_templates/539) in case you want a cool badge for LinkedIn.

You don't need to watch all of these videos or read all of the material, but having a basic understanding of how LLMs work is beneficial for training them effectively.


## Training LLMs

Training LLMs involves several intricate steps, each crucial for developing models that can effectively understand and generate human-like text. This document outlines these steps, focusing particularly on the methods applicable to our work, and provides insights into the tools and techniques commonly used in this domain.

## Steps of Training

Training a modern LLM happens in multiple phases, each with a different type of data, objective, and optimization strategy.

### Pretraining on a Large Corpus

Pretraining involves training the language model on a vast, diverse corpus of text. This stage aims to help the model learn a broad understanding of language, including syntax, grammar, and a bit of world knowledge. The model is trained to predict the next word in a sentence, helping it to learn contextual relationships between words.

Pretraining is the longest and most expensive phase of training. The produced model is commonly referred to as a **foundation model** or **pretrained model**. All it does is predict the next words, given a sequence of text. These models have no conception of roles, messages, etc., and are not directly useful for any specific task.

### Instruction Tuning

After pretraining, instruction tuning (or prompt tuning) tailors the model to follow specific instructions or prompts. This step requires a more manicured dataset, where each example has to have the desired format. In particular, it introduces the difference between **context**** and **instruction**, as well as between **user input** and **model output**.

This phase yields models that are typically called **instruction-tuned models** (or **xxx-instruct**, **xxx-chat**). Both foundation- and instruction-tuned models are available for most open-source models, such as Llama or Mixtral.


### Reinforcement Learning from Human Feedback (RLHF)

In RLHF, the model is fine-tuned based on human feedback to align its outputs more closely with human preferences. This approach involves humans rating model-generated answers, and the model is trained to optimize these ratings, improving its reliability and appropriateness.
The datasets used in RLHF are typically smaller, as they are very hard to generate and are kept private by companies like OpenAI.


### LORA Tuning

A lot of use cases require domain-specific fine-tuning of LLMs, e.g., in medicine, law, etc. While one can fine-tune the entire model on a small dataset, this is often not feasible due to the computational cost and the risk of overfitting. **LoRA tuning** is a method that allows for fine-tuning only a small subset of the model's parameters, reducing the computational cost and preserving much of the original model's structure and knowledge.
Instead of updating all parameters, LoRA focuses on modifying only a small subset, reducing the computational cost and preserving much of the original model's structure and knowledge. This method is especially useful for adapting large models to specific tasks without the need for extensive retraining.

## Quantization

Quantization is the process of reducing the precision of the model's numbers, which can decrease the model size and speed up inference times without significantly affecting performance. By default, most models are trained using 32-bit floating-point numbers. For a 8B-model, this means $8 \times 2^{30} \times 32b = 32GB$ of uncompressed data. If, instead, one only uses 4bit (called Q4-Quantization) integers, the model size is reduced to $8 \times 2^{30} \times 4b = 8GB$, which makes it feasible to put the model into RAM on most machines, and, with a bit of compression, even on phones.

Quantization and LoRA can be combined in a process called **Quantized LoRA**, or QLoRA, which allows for fine-tuning a quantized model on a small dataset.

## Common Libraries

### Hugging Face Transformers

One of the most popular libraries for working with large language models is Hugging Face's Transformers. This library provides a vast array of prebuilt models and tools for natural language processing tasks, making it easier to implement and fine-tune LLMs.

The following is a very basic example of how to fine-tune Meta's Llama2 model using Hugging Face's Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('facebook/llama-2')
model = AutoModelForCausalLM.from_pretrained('facebook/llama-2')

# Load datasets
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',            # output directory
    num_train_epochs=3,                # number of training epochs
    per_device_train_batch_size=8,     # batch size for training
    per_device_eval_batch_size=32,     # batch size for evaluation
    warmup_steps=500,                  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                 # strength of weight decay
    logging_dir='./logs',              # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",       # evaluate every logging_steps
    save_strategy="steps",             # save model every logging_steps
    save_total_limit=2                 # only keep the 2 most recent model checkpoints
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start training
trainer.train()
```

A more complete guide can be found in the [Hugging Face documentation](https://huggingface.co/transformers/training.html), including a detailed explanation of how to [train and fine-tune Llama2 using LoRA](https://huggingface.co/docs/trl/main/en/using_llama_models).


### Ollama

[Ollama](https://ollama.com) is a library that allows users to run open-source LLMs on their hardware efficiently and without any complicated setup. If you have never worked with LLMs offline and want to gauge their capabilities, Ollama is a great starting point.

Mac and Linux are currently supported in stable versions; Windows support is under preview (but seems to work quite well). If you have a recent machine, particularly a Mac with an M1 chip or later, you should be able to run 7B / 8B models at reasonable speed. For older machines or non-Mac hardware, 2B models might be the better starting point. Be sure to run quantized versions of the model in any case.

