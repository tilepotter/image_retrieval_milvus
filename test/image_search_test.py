import json

from pymilvus import connections, FieldSchema, DataType, CollectionSchema, Collection, Milvus

import milvus.milvus_util as milvusUtil
from oss.oss_util import MinioClient


def create_collection(collection_name):
    # 字段结构
    file_name = FieldSchema(
        name="file_name",
        dtype=DataType.VARCHAR,
        max_length=255,
        is_primary=True
    )

    feature_vector = FieldSchema(
        name="feature_vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=512
    )

    # schema 结构
    schema = CollectionSchema(
        fields=[file_name, feature_vector],
        description="image feature vector",
        enable_dynamic_field=False
    )
    # 创建集合
    return milvusUtil.create_collection(schema, collection_name)


def insert_data(img_path, collection_name):
    vgg_url = 'http://{0}:{1}/imageFeatureVector/calFeatureVector'.format('127.0.0.1', "5004")
    feature_vector = milvusUtil.extract_feature_vector_by_path(img_path, vgg_url)
    # image_bean = searchBean.ImageSearchBean(img_path, vector_data)
    rows_data = [
        {"file_name": img_path, "feature_vector": feature_vector}
    ]
    print(f'rows_data: ', rows_data)
    # 往milvus插入数据
    milvusUtil.insert_data(rows_data, collection_name)


def create_index(collection_name, index_field):
    index_params = {
        "metric_type": "L2",  # L2（欧几里得距离）
        "index_type": "IVF_FLAT",  # 浮点类型向量NewIndexIvfFlat
        "params": {"nlist": 1024}
    }
    # 创建索引
    milvusUtil.create_index(index_params, collection_name, index_field)


def vector_search(img_path, collection_name):
    # Milvus 内的所有搜索和查询操作都在内存中执行。在进行向量相似性搜索之前将集合加载到内存中。
    collection = Collection(collection_name)  # Get an existing collection.
    collection.load()
    # 查询参数
    search_params = {
        "metric_type": "L2",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }
    vgg_url = 'http://{0}:{1}/imageFeatureVector/calFeatureVector'.format('127.0.0.1', "5004")
    vector_data = milvusUtil.extract_feature_vector_by_path(img_path, vgg_url)
    results = collection.search(
        data=[vector_data],
        # 要搜索的字段名称。
        anns_field="feature_vector",
        # the sum of `offset` in `param` and `limit`
        # should be less than 16384.
        param=search_params,
        # 返回的最相似结果的数量。
        limit=5,
        expr=None,
        # set the names of the fields you want to
        # retrieve from the search result.
        output_fields=['file_name'],
        consistency_level="Strong"
    )
    # 释放内存
    collection.release()
    print(results)
    print(results[0].distances)


if __name__ == '__main__':
    # 创建milvus连接
    conn = connections.connect(
        alias="default",
        host="192.168.31.237",
        port="19530"
    )
    collection_name = "image_search"
    # 创建集合
    # 插入数据到集合
    # insert_data("D://Download//Train_Images_Set//set01_500//2007_000027.jpg", collection_name)
    # 创建索引
    # create_index(collection_name, "feature_vector")

    # 检索
    # milvus("D://Download//Train_Images_Set//set01_500//2007_000027.jpg", collection_name)
    # 封装方法测试
    vgg_url = 'http://{0}:{1}/imageFeatureVector/calFeatureVector'.format('127.0.0.1', "5004")
    img_feature_vector = milvusUtil.extract_feature_vector_by_path(
        "D:/Download/Train_Images_Set/set01_500/2007_000061.jpg", vgg_url)
    result = milvusUtil.search(collection_name="image_feature_vector", page=1, page_size=10,
                               similarity=0.85, img_feature_vector=img_feature_vector, search_field="feature_vector")
    print(result)
