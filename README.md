# image_retrieval_milvus
- **项目简介**：以图搜图后端服务实现。使用`milvus`向量引擎数据库存储图片的特征向量， 并使用`pymilvus`（milvus的python sdk）操作`milvus`进行创建集合、插入向量数据、构建向量索引、向量检索召回。使用`Flask`封装`HTTP RESTful`的相似度检索接口。
- **技术框架**：

| 向量存储引擎 | Milvus            |
| ------------ | ----------------- |
| 图片存储     | 开源对象存储MinIO |
| Web框架      | Flask             |



#  快速开始

## 1. docker-compose快速部署 Milvus 、MinIO：

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

相对`Milvus`官网提供的`docker-compose.yml`配置，增加了一个 `attu`服务，是向量存储库的`Web`可视化服务端。



## 2. 启动服务：

```she
docker-compose  up -d
```

访问`attu web` 可视化界面：`http://ip:8001/`，可看到如下界面：

![img](D:\IdeaWorkSpace\pycharm-project\image_retrieval_milvus\assert\img.png)



## 3. 安装项目依赖：

```pyth
pip install -r requirements.txt
```



## 4. 初始化数据

修改`milvus/save_feature_vector_to_milvus.py`文件中的配置信息，包括`Milvus`的 ip 和 端口，对象存储的相关配置，以及需要初始化的图片集路径：

![img_1](D:\IdeaWorkSpace\pycharm-project\image_retrieval_milvus\assert\img_1.png)

运行该 python 文件，会调用 vgg 模型后端服务接口，提取指定路径下的图片集的特征向量，并将特征向量写入 Milvus，图片本身存储至  MinIO：

```python
python3.8 milvus/save_feature_vector_to_milvus.py
```



## 5. 检索测试

运行 `server.py`，启动`Web`程序：

```py
python3.8 server.py
```

使用 postman 测试检索接口如下：

![img_2](D:\IdeaWorkSpace\pycharm-project\image_retrieval_milvus\assert\img_2.png)

请求参数释义：

* img_base64：图片的base64字符串
* similarity：相似度，支持 0.0 ~ 1.0 之间。项目对相似度和 Milvus 检索出的向量距离进行了转换，转换逻辑：similarity = 1 / (1 + hit.distance)
* page：页码
* page_size：每页条数

响应参数释义：

* file_name：文件名
* id：图片向量在milvus中存储的主键id
* img_url：图片在对象存储上的临时访问url
* similarity：相似度
