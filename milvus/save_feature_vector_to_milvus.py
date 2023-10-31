import logging
import os
import uuid

from pymilvus import FieldSchema, DataType, CollectionSchema, connections

import oss.oss_util as oss_util
import milvus.milvus_util as milvus_util

# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_collection(collection_name):
    """
    1.创建集合
    :param collection_name: 集合名称（相当于关系型数据库的表名）
    :return:
    """
    # 定义字段结构
    id = FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        max_length=255,
        is_primary=True
    )
    file_name = FieldSchema(
        name="file_name",
        dtype=DataType.VARCHAR,
        max_length=255,
    )
    feature_vector = FieldSchema(
        name="feature_vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=512
    )
    schema = CollectionSchema(
        fields=[id, file_name, feature_vector],
        description="image feature vector",
        enable_dynamic_field=False
    )
    logger.info("创建milvus集合 %s ...", collection_name)
    milvus_util.create_collection(schema, collection_name)


def save_data(collection_name, img_set_path, bucket_name, oss_host, oss_access_key, oss_secret_key):
    """
    初始化数据
    :param collection_name: 集合名
    :param img_set_path: 图片集路径
    :param bucket_name: OSS桶名
    :param oss_host: 对象存储主机
    :param oss_access_key: 对象存储 access_key
    :param oss_secret_key: 对象存储 secret_key
    :return:
    """
    vgg_url = 'http://{0}:{1}/imageFeatureVector/calFeatureVector'.format('127.0.0.1', "5004")
    img_path_list = milvus_util.find_filepaths(img_set_path)
    rows_data = []
    # 获取对象存储客户端
    client = oss_util.MinioClient(service=oss_host, access_key=oss_access_key, secret_key=oss_secret_key)
    for img_path in img_path_list:
        # 获取每一张图片的特征向量
        img_vector = milvus_util.extract_feature_vector_by_path(img_path, vgg_url)
        row = {'id': str(uuid.uuid4()), 'file_name': os.path.basename(img_path), 'feature_vector': img_vector}
        rows_data.append(row)
        # 每一张图片上传
        logger.info("上传图片 {} 至MinIO...".format(img_path))
        client.upload_file(bucket_name, os.path.basename(img_path), img_path, content_type="application/octet-stream")
    # 数据入milvus
    logger.info("插入数据集至Milvus,数据集大小：%d ...", len(rows_data))
    milvus_util.insert_data(rows_data, collection_name)


def create_index(collection_name, index_field):
    """
    创建索引
    :param collection_name: 集合名称
    :param index_field: 索引字段
    :return:
    """
    # 索引参数
    index_params = {
        "metric_type": "L2",  # L2（欧几里得距离）
        "index_type": "IVF_FLAT",  # 浮点类型向量NewIndexIvfFlat
        "params": {"nlist": 1024}
    }
    # 创建索引
    logger.info("正在创建集合 %s 的 %s 字段索引...", collection_name, index_field)
    milvus_util.create_index(index_params, collection_name, index_field)


if __name__ == '__main__':
    # 创建milvus连接
    conn = connections.connect(
        alias="default",
        host="192.168.31.237",
        port="19530"
    )
    # 对象存储主机
    oss_host = "192.168.31.237:9002"
    # 对象存储 access_key
    oss_access_key = "admin"
    # 对象存储 secret_key
    oss_secret_key = "admin123456"

    # 存储桶名
    bucket_name = "image-retrieval"
    # 集合名
    collection_name = "image_feature_vector"
    # 创建集合
    create_collection(collection_name)

    # 插入数据
    save_data(collection_name=collection_name, img_set_path="D://Download//Train_Images_Set//set02_7",
              bucket_name=bucket_name, oss_host=oss_host, oss_access_key=oss_access_key, oss_secret_key=oss_secret_key)

    # 创建索引
    create_index(collection_name, "feature_vector")
