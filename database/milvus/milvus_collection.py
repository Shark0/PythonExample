from pymilvus import connections, utility, Collection
from pymilvus.exceptions import MilvusException

# 連接到 Milvus
try:
    connections.connect(host="192.168.6.70", port="19530")
    print("Connected to Milvus!")
except MilvusException as e:
    print(f"Failed to connect to Milvus: {e}")
    exit(1)

# 列出所有 collection 並顯示字段和索引
try:
    collections = utility.list_collections()
    print(f"Total number of collections: {len(collections)}")

    if not collections:
        print("No collections found.")
    else:
        print("Details of collections:")
        for i, collection_name in enumerate(collections, 1):
            try:
                collection = Collection(collection_name)
                print(f"\n{i}. Collection: {collection_name}")

                # 獲取字段信息
                schema = collection.schema
                print("   Fields:")
                for field in schema.fields:
                    field_info = f"      - Name: {field.name}, Type: {field.dtype.name}"
                    if field.is_primary:
                        field_info += f", Primary Key: {field.is_primary}, Auto ID: {field.auto_id}"
                    if field.params:
                        field_info += f", Params: {field.params}"
                    print(field_info)

                # 獲取索引信息
                indexes = collection.indexes
                print("   Indexes:")
                if indexes:
                    for index in indexes:
                        try:
                            index_params = index.params
                            index_type = index_params.get('index_type', 'Unknown')
                            metric_type = index_params.get('metric_type', 'Unknown')
                            params = index_params.get('params', {})
                            print(f"      - Field: {index.field_name}, Type: {index_type}, "
                                  f"Metric: {metric_type}, Params: {params}")
                        except Exception as e:
                            print(f"      - Error retrieving index details for {index.field_name}: {e}")
                else:
                    print("      No indexes found.")
            except MilvusException as e:
                print(f"Error retrieving details for collection {collection_name}: {e}")
except MilvusException as e:
    print(f"Error listing collections: {e}")
    exit(1)