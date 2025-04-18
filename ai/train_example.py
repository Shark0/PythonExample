from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
# 1. Load a model to finetune
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# 2. Load the dataset for fine-tuning, omitting pre-processing
dataset = load_dataset("csv", data_files="training_data.csv", encoding="utf-8", split="train")
print(dataset)
# Check
for row in dataset:
    print(row)
# 3. Define a loss function
loss = MultipleNegativesRankingLoss(model)
# The sentences to encode
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
# tensor([[0.3030, 0.4355, 1.0000],
#         [0.1266, 0.2702, 0.7639],
#         [0.4968, 0.6839, 0.4712],
#         [0.1700, 0.1781, 0.2294],
#         [0.1816, 0.4446, 0.7958]])
# 4. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=dataset,
    loss=loss
)
trainer.train()
# After training
print("\nAfter training")
label_embeddings = model.encode(labels)
input_embeddings = model.encode(inputs)
print(label_embeddings.shape)
print(input_embeddings.shape)
similarities = model.similarity(input_embeddings, label_embeddings)
print(similarities)
# tensor([[ 0.0811, -0.0096,  1.0000],
#         [-0.0107,  0.0396,  0.8212],
#         [ 0.3265,  0.8719, -0.1230],
#         [ 0.1643,  0.1345,  0.1480],
#         [ 0.0728,  0.3322,  0.7165]])
# 5. Save the trained model
model.save_pretrained("Model_001")