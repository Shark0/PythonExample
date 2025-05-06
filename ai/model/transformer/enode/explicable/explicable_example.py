import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from bertviz import head_view
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

# 設置模型和數據集
model_name = "sentence-transformers/all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(model_name)
hf_model = AutoModel.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加載 STS-Benchmark 數據集
dataset = load_dataset("glue", "stsb", split="validation")

# 選擇一個句子對
sentence1 = dataset[0]["sentence1"]  # 例如: "A man is playing a guitar."
sentence2 = dataset[0]["sentence2"]  # 例如: "A person is strumming a guitar."
label = dataset[0]["label"]

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Similarity Score: {label}")

# 分詞並提取注意力分數
inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs1 = hf_model(**inputs1)
    outputs2 = hf_model(**inputs2)

attentions1 = outputs1.attentions  # (num_layers, batch_size, num_heads, seq_len, seq_len)
tokens1 = tokenizer.convert_ids_to_tokens(inputs1["input_ids"][0])

# 可視化第一層第一個注意力頭
layer, head = 0, 0
attention1 = attentions1[layer][0][head].numpy()

# 使用 BertViz 可視化（註：在 Jupyter 中運行以顯示熱圖）
head_view([attention1], tokens1, layer=layer, heads=[head])

# 語義一致性分析：計算關鍵詞的注意力分數總和
guitar_idx = tokens1.index("guitar") if "guitar" in tokens1 else -1
if guitar_idx != -1:
    attention_sum = attention1[guitar_idx].sum()
    print(f"Attention sum for 'guitar': {attention_sum}")
    for i, token in enumerate(tokens1):
        print(f"Token: {token}, Attention Sum: {attention1[i].sum()}")

# 計算原始句子相似度
embeddings = sentence_model.encode([sentence1, sentence2])
cosine_sim = sentence_model.similarity(embeddings[0], embeddings[1]).item()
print(f"Original Cosine Similarity: {cosine_sim}")

# 介入實驗：屏蔽第一層第一個注意力頭
def mask_attention_head(model, layer, head):
    model.encoder.layer[layer].attention.self.attention_probs.data[:, head, :, :] = 0

with torch.no_grad():
    mask_attention_head(hf_model, layer=0, head=0)
    outputs1 = hf_model(**inputs1)
    outputs2 = hf_model(**inputs2)
    embedding1 = outputs1.last_hidden_state.mean(dim=1).squeeze().numpy()
    embedding2 = outputs2.last_hidden_state.mean(dim=1).squeeze().numpy()
    masked_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    print(f"Masked Cosine Similarity: {masked_sim}")