from oss.oss_util import MinioClient

"""
对象存储上传、下载测试
"""

if __name__ == '__main__':
    client = MinioClient(service="192.168.31.237:9002", access_key="admin", secret_key="admin123456")
    # 查询所有存储桶
    buckets = client.get_bucket_list()
    print(buckets)

    # 创建存储桶
    # is_create = client.create_bucket(bucket_name="image-retrieval", is_policy=False)
    # print(f'创建存储桶是否成功：', is_create)

    # 查看桶是否存在
    is_exists = client.exists_bucket(bucket_name="image-retrieval")
    print(f'桶是否存在', is_exists)

    # 上传文件
    # client.upload_file(bucket_name="image-retrieval", file="2007_000032.jpg",
    #                    file_path='D://Download//Train_Images_Set//set01_500//2007_000032.jpg',
    #                    content_type="application/octet-stream")

    # 下载文件
    client.download_file(bucket_name="image-retrieval", file="2007_000032.jpg",
                         file_path="C://Users//lenovo//Desktop//test.jpg")
