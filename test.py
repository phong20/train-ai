 # -*- coding: utf-8 -*-
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model_name = 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
input_text = "dịch đoạn văn này sang tiếng việt: 寿命を買い取ってもらえるという話を聞いたとき、俺が真っ先に思い出したのは、小学生の頃に受けた道徳の授業のことだった。まだ自分でものを考えるということを知らない十歳の俺たちに向けて、学級担任である二十代後半の女性教員は、こんなふうに問いかけた。"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=1000000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
