import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

df = pd.read_csv('dataset.csv')

texts = df['vietnamese'].tolist() + df['japanese'].tolist()

dataset = Dataset.from_dict({'text': texts})

model_name = 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

model = GPT2LMHeadModel.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
