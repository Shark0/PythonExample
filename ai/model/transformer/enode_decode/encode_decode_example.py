import torch
from transformers import MarianTokenizer, MarianMTModel
import transformers

if __name__ == "__main__":
    # 使用支援多語言的 M2M100 模型
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )

    # 解碼翻譯文字
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"Translated text: {translated_text}")