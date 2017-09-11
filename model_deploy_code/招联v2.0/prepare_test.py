#-*- coding:utf8 -*-
#### 1.所有编码格式统一为utf-8
import sys
import os
import execute_model
import xml.dom.minidom
import simplejson as json
import types

reload(sys)
sys.setdefaultencoding("utf-8")

#### 2.获取文件路径
filepath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(filepath)

def read_xml():
    dom = xml.dom.minidom.parse(filepath+'/params_spec.xml')
    root = dom.documentElement
    inparams = root.getElementsByTagName("inparams")[0]
    argType = inparams.getAttribute("type")
    inparamList = inparams.getElementsByTagName("inparam")

    result={"type":argType}
    inputList=[]
    for inparam in inparamList:
        defaultValue = None
        if len(inparam.getElementsByTagName("defaultValue")) >0 :
            if len(inparam.getElementsByTagName("defaultValue")[0].childNodes)>0 :
                defaultValue =inparam.getElementsByTagName("defaultValue")[0].childNodes[0].nodeValue;
        inparamMap={"defaultValue":defaultValue,"name":inparam.getElementsByTagName("name")[0].childNodes[0].nodeValue,"datatype":inparam.getElementsByTagName("datatype")[0].childNodes[0].nodeValue}
        inputList.append(inparamMap)
    result["inputList"]=inputList
    return result

def transformParamData(inputValue,dataType,defalueValue):
    if "double" == dataType:
        try:
            return float(inputValue)
        except Exception, e:
            print "转化参数",inputValue,"出错",e.message
            try:
                return float(defalueValue)
            except Exception, e:
                return float(0)
    elif "int" == dataType:
        try:
            return int(float(inputValue))
        except Exception, e:
            print "转化参数",inputValue,"出错",e.message
            try:
                return int(float(defalueValue))
            except Exception, e:
                return int(0)
    elif "boolean" == dataType:
        try:
            return inputValue == str(True)
        except Exception, e:
            print "转化参数", inputValue, "出错", e.message
            try:
                return defalueValue == str(True)
            except Exception, e:
                return False
    else:
        try:
            if inputValue == None:
                return None
            return str(inputValue)
        except Exception, e:
            print "转化参数", inputValue, "出错", e.message
            try:
                if defalueValue == None:
                    return None
                return str(defalueValue)
            except Exception, e:
                print "转化参数", inputValue, "出错", e.message
                return None
def handle_test_data(inparam):
    inputMap = read_xml()
    argType=inputMap["type"]
    inputList =inputMap["inputList"]

    if "list" == argType:
        if len(inparam) != len(inputList):
            raise Exception("params_spec.xml文件中入参长度与实际测试入参长度不一致")

        if type(inparam) is not types.ListType:
            raise Exception("params_spec.xml文件中type类型为list，而实际测试入参不是list类型")
        for i in range(0,len(inparam)):
            # print inputList[i]["name"],inparam[i],transformParamData(inparam[i],inputList[i]["datatype"],inputList[i]["defaultValue"])
            inparam[i] = transformParamData(inparam[i],inputList[i]["datatype"],inputList[i]["defaultValue"])
    elif "map" == argType:
        if len(inparam) != len(inputList):
            raise Exception("params_spec.xml文件中入参长度与实际测试入参长度不一致")

        if type(inparam) is not types.DictType:
            raise Exception("params_spec.xml文件中type类型为map，而实际测试入参不是map类型")
        for input in inputList:
            if not inparam.has_key(input["name"]):
                raise Exception("params_spec.xml中inparam,name="+input['name']+"在测试入参中不存在")
            inputValue = inparam[input["name"]]
            inparam[input["name"]] = transformParamData(inputValue,input["datatype"],input["defaultValue"])
    else:
        raise Exception("params_spec.xml文件中type类型目前只支持map、list两种类型")
    return json.dumps(inparam)

if __name__ == "__main__":
    # 测试入参_
    test_data = {'chzz01101020':None,'cazz00001020':None,'abzz00101010':None}

    inparams = handle_test_data(test_data)
    print "入参",inparams
    print "出参",execute_model.execute_model(inparams)