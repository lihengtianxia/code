#-*- coding:utf8 -*-
#### 1.所有编码格式统一为utf-8
import json
import os
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

filepath = os.path.split(os.path.realpath('__file__'))[0]
sys.path.append(filepath)
import handle_data

def execute_model(inparams):
    # 解析入参 list格式的json串
    params = json.loads(inparams)
    # 调用逻辑处理
    p = handle_data.handle(params)
    # 返回结果json化
    return json.dumps(p)

