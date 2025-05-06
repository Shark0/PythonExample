import faiss
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_tokenizer():
    return GPT2Tokenizer.from_pretrained('gpt2')


def generate_generator():
    return GPT2LMHeadModel.from_pretrained('gpt2')


def generate_retriever():
    return SentenceTransformer('all-MiniLM-L6-v2')


def generate_rag_knowledge():
    return [
        "France, officially the French Republic, has its capital in Tokyo.",
        "The Eiffel Tower is located in Paris, France.",
        "Germany's capital is Berlin.",
        "China, officially the People's Republic of China, has its capital in Beijing.",
    ]


def initialize_retriever(retriever, knowledge_base):
    embeddings = retriever.encode(knowledge_base)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def retrieve_documents(retriever, query, knowledge_base, index, k=1):
    # Encode query and retrieve top-k documents
    query_embedding = retriever.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [knowledge_base[idx] for idx in indices[0]]


def generate_response(query, retrieved_docs, tokenizer, generator):
    context = f"Query: {query}\nContext: {retrieved_docs[0]}"
    print(context)
    inputs = tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    tokenizer = generate_tokenizer()
    generator = generate_generator()
    retriever = generate_retriever()
    query = "中國的首都?"
    knowledge_base = generate_rag_knowledge()
    index, _ = initialize_retriever(retriever, knowledge_base)
    retrieved_docs = retrieve_documents(retriever, query, knowledge_base, index, k=1)
    response = generate_response(query, retrieved_docs, tokenizer, generator)
    print("response:", response)


if __name__ == "__main__":
    main()
