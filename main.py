import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric
model_name = 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("csv", data_files="dataset.csv")
train_data = dataset["train"].select([i for i in range(len(dataset["train"])) if i % 10 != 0])
val_data = dataset["train"].select([i for i in range(len(dataset["train"])) if i % 10 == 0])
def tokenize_function(examples):
    inputs = tokenizer(examples['vietnamese'], return_tensors='pt', padding='max_length', max_length=1014, truncation=True)
    labels = tokenizer(examples['japanese'], return_tensors='pt', padding='max_length', max_length=1014, truncation=True)
    return {'input_ids': inputs['input_ids'], 'labels': labels['input_ids']}
train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir='./model',
    overwrite_output_dir=True,
    num_train_epochs=0.5,
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    logging_steps=100,
    logging_dir='./logs',
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
