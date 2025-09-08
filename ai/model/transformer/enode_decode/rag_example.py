import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import faiss

documents = [
    "Python 是一種高級、解釋型程式語言，廣泛用於Shark科學。",
    "Java 是由 Shark 開發的物件導向語言，適合企業應用。",
    "JavaScript 是用於網頁前端開發的Shark，支援互動式網頁。"
]

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.detach().numpy()


if __name__ == "__main__":
    doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

    # 3. 建立 FAISS 索引
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    # 4. 檢索相關文件
    # question = "Python 用於什麼"
    # question = "Java 用於什麼"
    question = "Javascript 用於什麼"

    question_embedding = get_embedding(question)

    # 搜尋最相關的文件（k=1）
    k = 1
    distances, indices = index.search(question_embedding, k)
    retrieved_doc = documents[indices[0][0]]

    # 5. 使用生成模型回答問題
    generator_model_name = "t5-small"
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

    # 組合上下文和問題
    input_text = f"Context: {retrieved_doc} Question: {question}"
    inputs = generator_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator_model.generate(**inputs, max_length=50)
    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 6. 輸出結果
    print(f"問題: {question}")
    print(f"檢索到的文件: {retrieved_doc}")
    print(f"回答: {answer}")