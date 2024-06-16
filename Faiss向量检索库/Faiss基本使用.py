import faiss
import numpy as np

'''
1. 基本操作：使用的线性搜索
'''


def basic():
    # 1. 创建索引（索引数据库）
    # 参数用于指定存储的向量维度
    index = faiss.IndexFlatL2(256)  # 线性搜索  L2 表示使用相似度计算的是：欧式距离
    index = faiss.IndexFlatIP(256)  # 线性搜索  IP 点击相似度
    # 工厂方法创建索引
    index = faiss.index_factory(256, 'Flat', faiss.METRIC_L2)
    index = faiss.index_factory(256, 'Flat', faiss.METRIC_INNER_PRODUCT)

    # 2. 添加索引
    vectors = np.random.rand(10000, 256)  # 创建10000个256维的向量
    # 将上面10000个向量添加到index索引库中，它会给每一个向量分配一个索引，索引是从0开始的（这种方法我们不能自己指定id，使用下面的ID映射）
    index.add(vectors)

    # 3. 搜索向量
    query = np.random.rand(2, 256)
    D, I = index.search(query, k=1)
    print(I)
    print(D)

    # 4. 删除向量
    index.remove_ids(np.array([1, 2, 3]))
    print(index.ntotal)

    index.reset()  # 删除全部向量
    print(index.ntotal)

    # 5. 存储向量（持久化）
    faiss.write_index(index, "vectors.faiss")

    # 6. 加载索引
    index = faiss.read_index("vectors.faiss")
    print(index)


'''
2. ID映射
'''


def id_test():
    # 1. 创建索引
    index = faiss.IndexFlatL2(256)
    # 包装索引，实现自定义向量ID
    index = faiss.IndexIDMap(index)

    # 2. 添加索引
    vectors = np.random.rand(10000, 256)
    # 人为指定，id从10000开始，到20000
    index.add_with_ids(vectors, np.arange(10000, 20000))
    print(index.ntotal)

    # 3. 搜索向量
    query = np.random.rand(2, 256)
    D, I = index.search(query, k=1)
    print(I)
    print(D)


if __name__ == '__main__':
    id_test()
