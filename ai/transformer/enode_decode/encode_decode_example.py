from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# 使用支援多語言的 M2M100 模型
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

# 設定語言對
tokenizer.src_lang = "en"
input_text = "Hello, how are you?"

# 將輸入文字轉為 token ids
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 翻譯成德文（de）
generated_tokens = model.generate(input_ids, forced_bos_token_id=tokenizer.get_lang_id("de"))

# 解碼翻譯文字
translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print("Translated text:", translated_text)