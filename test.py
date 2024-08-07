 # -*- coding: utf-8 -*-
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model_name = 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
input_text = "Hãy viết một đoạn văn bản tiếp theo từ đây:"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
