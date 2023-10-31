# -*- coding: utf-8 -*-
import base64
import json
import os

import requests
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

from oss.oss_util import MinioClient


def create_collection(collection_schema, collection_name):
    """
    创建集合
    :param collection_schema:  集合结构
    :param collection_name:  集合名称
    :return: 是否创建成功
    """

    collection = Collection(
        name=collection_name,
        schema=collection_schema,
        using='default',
        shards_num=2,
    )
    # 查看集合是否存在
    return utility.has_collection(collection_name)


def insert_data(data, collection_name):
    """
    对指定集合插入数据
    :param data:  需要插入的数据
    :param collection_name:  集合名称
    :return:
    """

    collection = Collection(collection_name)
    collection.insert(data)
    collection.flush()


def bulk_data(collection_name, file):
    collection = Collection(collection_name)
    utility.do_bulk_insert(collection_name, file)


def create_index(index_params, collection_name, index_field):
    """
    创建索引
    :param index_params:  索引参数
    :param collection_name: 集合名称
    :param index_field: 索引字段
    :return:
    """

    collection = Collection(collection_name)
    collection.create_index(
        field_name=index_field,
        index_params=index_params
    )

    # 构建索引
    utility.index_building_progress(collection_name)


def drop_index(collection_name):
    """
     删除索引
    :param collection_name: 集合名称
    :return:
    """

    collection = Collection(collection_name)
    collection.drop_index()


def drop_collection(collection_name):
    """
     删除集合
    :param collection_name: 集合名称
    :return:
    """

    collection = Collection(collection_name)
    collection.drop()


def drop_partition(collection_name, partition):
    """
    删除分区
    :param collection_name: 集合名称
    :param partition: 分区名
    :return:
    """
    collection = Collection(collection_name)
    collection.drop_partition(partition_name=partition)


def extract_feature_vector_by_path(img_path, vgg_url):
    """
    调用VGG算法提取图片特征向量_通过图片路径
    :param img_path: 图片路径
    :param vgg_url: vgg接口url
    :return: 图片的特征向量集合
    """
    f = open(img_path, 'rb')
    # base64编码
    base64_data: bytes = base64.b64encode(f.read())
    f.close()
    base64_data = base64_data.decode()
    # 传输的数据格式
    data = {'img': base64_data}
    # post传递数据
    headers = {'Content-Type': 'application/json'}
    r = requests.post(vgg_url, headers=headers, data=json.dumps(data))
    return json.loads(r.text)['data']


def extract_feature_vector_by_base64(img_base64, vgg_url):
    """
    调用VGG算法提取图片特征向量_通过图片base64字符串
    :param img_base64: 图片base64字符串
    :param vgg_url:  vgg接口url
    :return:
    """
    # 传输的数据格式
    data = {'img': img_base64}
    # post传递数据
    headers = {'Content-Type': 'application/json'}
    r = requests.post(vgg_url, headers=headers, data=json.dumps(data))
    return json.loads(r.text)['data']


def search(collection_name, page, page_size, similarity, img_feature_vector, search_field):
    """
    向量检索
    :param collection_name: 集合名
    :param page: 页码
    :param page_size:  条数
    :param similarity:  相似度
    :param img_feature_vector: 图片特征向量
    :param search_field:  检索字段
    :return:
    """
    # Milvus 内的所有搜索和查询操作都在内存中执行。在进行向量相似性搜索之前将集合加载到内存中。
    collection = Collection(collection_name)  # Get an existing collection.
    collection.load()
    # 请求分页计算
    offset = ((page - 1) * page_size) if (page > 0) else 0
    limit = page_size * (1 if (page > 0) else 0)

    # 查询参数
    search_params = {
        "metric_type": "L2",  # 欧氏距离
        "offset": offset,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }
    # 相似度检索
    search_result = collection.search(
        data=[img_feature_vector],
        # 要搜索的字段名称。
        anns_field=search_field,
        # the sum of `offset` in `param` and `limit`
        # should be less than 16384.
        param=search_params,
        # 返回的最相似结果的数量。
        limit=limit,
        expr=None,
        # set the names of the fields you want to
        # retrieve from the search result.
        output_fields=['file_name'],
        consistency_level="Strong"
    )
    # 释放内存
    # collection.release()
    # 处理返回结果
    hits = search_result[0]
    oss_client = get_minio_client()
    rows = []
    for hit in hits:
        # 将距离转为相似度
        obj_similarity = 1 / (1 + hit.distance)
        img_url = oss_client.presigned_get_file("image-retrieval", hit.entity.get("file_name"))
        # 相似度过滤
        if obj_similarity >= similarity:
            row = {"id": hit.id, "similarity": obj_similarity, "file_name": hit.entity.get("file_name"),
                   "img_url": img_url}
            rows.append(row)
    # 也可在外层使用 lambda 过滤list
    # filter_rows = list(filter(lambda item: item['similarity'] >= similarity, rows))
    return rows


def get_minio_client():
    """
    获取对象存储客户端连接
    :return:
    """
    return MinioClient(service="192.168.31.237:9002", access_key="admin", secret_key="admin123456")


def find_filepaths(dir):
    """
    级联遍历目录，获取目录下的所有文件路径
    :param dir:  指定目录
    :return:
    """
    result = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            filepath = os.path.join(root, name)
            if os.path.exists(filepath):
                result.append(filepath)
    return result
