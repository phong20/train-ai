from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Tải mô hình GPT-2 XL và tokenizer
model_name = 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Sinh văn bản
input_text = "Hãy viết một đoạn văn bản tiếp theo từ đây:"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Sinh văn bản
output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Chuyển đổi đầu ra thành văn bản
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
