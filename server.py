from flask import Flask, jsonify
from flask_cors import CORS
from pre_request import pre, Rule
from pymilvus import connections

import milvus.milvus_util as milus_util
from api.code import ResponseCode, ResponseMessage
from api.log import logger

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/imageRetrieval/search', methods=['post'])
def imageFeatureVector():
    """
    图片相似度检索接口
    :return:
    """
    # 参数校验并获取参数
    rule = {
        "img_base64": Rule(type=str, required=True),
        "similarity": Rule(type=float, require=True),
        "page": Rule(type=int, require=True, default=1),
        "page_size": Rule(type=int, require=True, default=10)
    }
    try:
        params = pre.parse(rule=rule)
        image_base64 = params.get("img_base64")
        similarity = params.get("similarity")
        page = params.get("page")
        page_size = params.get("page_size")
    except Exception as e:
        logger.error(e)
        fail_response = dict(code=ResponseCode.PARAM_FAIL, msg=ResponseMessage.PARAM_FAIL, data=None)
        logger.error(fail_response)
        return jsonify(fail_response)

    # 计算图片向量并进行相似度检索
    try:
        # 创建milvus连接
        conn = connections.connect(
            alias="default",
            host="192.168.31.237",
            port="19530"
        )
        vgg_url = 'http://{0}:{1}/imageFeatureVector/calFeatureVector'.format('127.0.0.1', "5004")
        img_feature_vector = milus_util.extract_feature_vector_by_base64(img_base64=image_base64, vgg_url=vgg_url)
        # 相似度检索
        result = milus_util.search(collection_name="image_feature_vector", page=page, page_size=page_size,
                                   similarity=similarity, img_feature_vector=img_feature_vector,
                                   search_field="feature_vector")
    except Exception as e:
        logger.error(e)
        fail_response = dict(code=ResponseCode.BUSINESS_FAIL, msg=ResponseMessage.BUSINESS_FAIL, data=None)
        logger.error(fail_response)
        return jsonify(fail_response)

    # 成功的结果返回
    success_response = dict(code=ResponseCode.SUCCESS, msg=ResponseMessage.SUCCESS, data=result)
    logger.info(success_response)
    return jsonify(success_response)


if __name__ == '__main__':
    # 解决中文乱码问题
    app.config['JSON_AS_ASCII'] = False
    # 启动服务 指定主机和端口
    app.run(host='0.0.0.0', port=5005, debug=False)
