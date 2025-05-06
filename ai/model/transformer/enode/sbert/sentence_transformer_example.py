from sentence_transformers import SentenceTransformer
import transformers
print(transformers.__version__)


import sys
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# The sentences to encode
ansSentences = [
    "退貨",
    "退货",
    "運費",
    "訂單查詢"
]
queSentences = [
    "如何辦理退貨？",
    "我要退貨",
    "退貨",
    "產品有問題，我要退貨",
    "想請教怎麼退貨?",
    "退货",
    "refund"
]
# 2. Calculate embeddings by calling model.encode()
ansEmbeddings = model.encode(ansSentences)
queEmbeddings = model.encode(queSentences)
print(ansEmbeddings.shape)
print(queEmbeddings.shape)
# 3. Calculate the embedding similarities
similarities = model.similarity(queEmbeddings, ansEmbeddings)
print(similarities)