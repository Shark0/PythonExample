from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from ai.model.transformer.enode.sbert.train_sentence_transformer_by_contrastive import get_trained_model

def get_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_dataset():
    dataset = load_dataset("csv", data_files="training_data.csv", encoding="utf-8", split="train")
    print(dataset)
    for row in dataset:
        print(row)
    return dataset

def classify(model):

    labels = [
        "如何查詢訂單狀態？",
        "運費是多少？",
        "如何辦理退貨？"
    ]
    inputs = [
        "如何辦理退貨？",
        "我要退貨",
        "我有一張訂單總額499元，需要支付多少運費",
        "今天星期幾",
        "你好商品有瑕疵, 我想連運費退款退貨"
    ]
    # Before training
    print("\nBefore training")
    label_embeddings = model.encode(labels)
    input_embeddings = model.encode(inputs)
    print(label_embeddings.shape)
    print(input_embeddings.shape)
    similarities = model.similarity(input_embeddings, label_embeddings)
    print(similarities)
    print("\n")

def train_model(model, dataset):
    loss = MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=dataset,
        loss=loss
    )
    trainer.train()
    model.save_pretrained("Model_001")

def get_trained_model():
    return SentenceTransformer("./Model_001")

def main():
    model = get_model()
    dataset = get_dataset()
    classify(model)
    train_model(model, dataset)
    trained_model = get_trained_model()
    classify(trained_model)

if __name__ == "__main__":
    main()