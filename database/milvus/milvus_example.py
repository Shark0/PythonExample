import os
from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import re

# 連接到 Milvus
connections.connect(host="192.168.6.70", port="19530")
print("Connected to Milvus!")

# 初始化 SentenceTransformer 模型（支援多語言，特別是中文）
model = SentenceTransformer("BAAI/bge-m3")

# 定義集合結構
collection_name = "mms_chat"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="milvus", dtype=DataType.FLOAT_VECTOR, dim=1024),  # BAAI/bge-m3 生成 1024 維向量
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096)
]
schema = CollectionSchema(fields=fields, description="MMS Q&A collection")
collection = Collection(name=collection_name, schema=schema)

# 創建索引
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64}
}
collection.create_index(field_name="milvus", index_params=index_params)
collection.load()
print(f"Collection {collection_name} created and loaded!")

# 讀取 docs 資料夾中的 qa.md 文件
docs_dir = Path("docs")
entries = []

for md_file in docs_dir.glob("*.md"):
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
        # 以 # 分割標題
        sections = re.split(r"#\s+(.+?)\n", content)[1:]  # 跳過開頭空內容
        for i in range(0, len(sections), 2):
            title = sections[i].strip()
            section_content = sections[i + 1].strip() if i + 1 < len(sections) else ""
            # 以 * 分割條列內容
            items = re.split(r"\*\s+", section_content)[1:]  # 跳過開頭空項
            for item in items:
                item = item.strip()
                if item:
                    entries.append({"title": title, "content": item})

# 生成向量並插入 Milvus
contents = [entry["content"] for entry in entries]
titles = [entry["title"] for entry in entries]

# 生成 content 的向量
vectors = model.encode(contents, convert_to_numpy=True).tolist()

# 準備插入數據
data = [
    vectors,  # milvus
    titles,   # title
    contents  # content
]

# 插入數據
collection.insert(data)
print(f"Inserted {len(entries)} entries into {collection_name}.")

# 查詢集合中的數據量
collection.load()
print(f"Total entities in {collection_name}: {collection.num_entities}")

# 多個搜索範例，針對 content 中的內容
search_queries = [
    "如何接收財務報告？",
    "誰可以更改用戶設定？",
    "如何查看繳款發票？"
]

for search_text in search_queries:
    print(f"\nSearching for: {search_text}")
    search_vector = model.encode([search_text])[0].tolist()
    search_params = {"metric_type": "L2", "params": {"ef": 10}}
    results = collection.search(
        data=[search_vector],
        anns_field="milvus",
        param=search_params,
        limit=3,
        output_fields=["title", "content"]
    )

    # 打印搜索結果
    for result in results[0]:
        print(f"Title: {result.entity.get('title')}")
        print(f"Content: {result.entity.get('content')}")
        print(f"Distance: {result.distance}")
        print("-" * 50)