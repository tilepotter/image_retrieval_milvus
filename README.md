# image_retrieval_milvus
- **��Ŀ���**����ͼ��ͼ��˷���ʵ�֡�ʹ��`milvus`�����������ݿ�洢ͼƬ������������ ��ʹ��`pymilvus`��milvus��python sdk������`milvus`���д������ϡ������������ݡ������������������������ٻء�ʹ��`Flask`��װ`HTTP RESTful`�����ƶȼ����ӿڡ�
- **�������**��

| �����洢���� | Milvus            |
| ------------ | ----------------- |
| ͼƬ�洢     | ��Դ����洢MinIO |
| Web���      | Flask             |



#  ���ٿ�ʼ

## 1. docker-compose���ٲ��� Milvus ��MinIO��

```dockerfile
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.2.10
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
  
  attu:
   container_name: milvus-attu
   image: zilliz/attu:v2.2.6
   environment:
     MILVUS_URL: standalone:19530
     HOST_URL: http://127.0.0.1:8001
   ports:
     - "8001:3000"
   depends_on:
     -  "standalone"
 
networks:
  default:
    name: milvus
```

���`Milvus`�����ṩ��`docker-compose.yml`���ã�������һ�� `attu`�����������洢���`Web`���ӻ�����ˡ�



## 2. ��������

```she
docker-compose  up -d
```

����`attu web` ���ӻ����棺`http://ip:8001/`���ɿ������½��棺

![img](D:\IdeaWorkSpace\pycharm-project\image_retrieval_milvus\assert\img.png)



## 3. ��װ��Ŀ������

```pyth
pip install -r requirements.txt
```



## 4. ��ʼ������

�޸�`milvus/save_feature_vector_to_milvus.py`�ļ��е�������Ϣ������`Milvus`�� ip �� �˿ڣ�����洢��������ã��Լ���Ҫ��ʼ����ͼƬ��·����

![img_1](D:\IdeaWorkSpace\pycharm-project\image_retrieval_milvus\assert\img_1.png)

���и� python �ļ�������� vgg ģ�ͺ�˷���ӿڣ���ȡָ��·���µ�ͼƬ��������������������������д�� Milvus��ͼƬ����洢��  MinIO��

```python
python3.8 milvus/save_feature_vector_to_milvus.py
```



## 5. ��������

���� `server.py`������`Web`����

```py
python3.8 server.py
```

ʹ�� postman ���Լ����ӿ����£�

![img_2](D:\IdeaWorkSpace\pycharm-project\image_retrieval_milvus\assert\img_2.png)

����������壺

* img_base64��ͼƬ��base64�ַ���
* similarity�����ƶȣ�֧�� 0.0 ~ 1.0 ֮�䡣��Ŀ�����ƶȺ� Milvus ���������������������ת����ת���߼���similarity = 1 / (1 + hit.distance)
* page��ҳ��
* page_size��ÿҳ����

��Ӧ�������壺

* file_name���ļ���
* id��ͼƬ������milvus�д洢������id
* img_url��ͼƬ�ڶ���洢�ϵ���ʱ����url
* similarity�����ƶ�
