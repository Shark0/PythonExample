import numpy as np


if __name__ == '__main__':
    num_users = 10  # 消費者數量
    num_articles = 20  # 文章數量
    vector_dim = 4  # Category Vector和Topic Vector的維度
    num_likes = 10  # 點讚紀錄數量

    user_vectors = np.random.rand(num_users, vector_dim).astype(np.float32)
    print('user_vectors: ', user_vectors)
    article_vectors = np.random.rand(num_articles, vector_dim).astype(np.float32)
    print('article_vectors: ', article_vectors)
    like_records = [(np.random.randint(0, num_users), np.random.randint(0, num_articles)) for _ in range(num_likes)]
    print('like_records: ', like_records)

    user_id = 0
    neg_article_id = np.random.randint(0, num_articles)
    while (user_id, neg_article_id) in set(like_records):  # 確保負樣本未被點讚
        neg_article_id = np.random.randint(0, num_articles)
    print('neg_article_id: ', neg_article_id)
