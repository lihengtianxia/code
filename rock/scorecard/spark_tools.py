# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
"""
从spark数据中根据iv，单一性等挑选出部分变量
在dc平台上请使用dc_spark2table函数
在堡垒机环境下请使用get_csv_from_spark函数
"""
from __future__ import unicode_literals, print_function, division
import pandas as pd
import numpy as np
from operator import add
import datetime
import json
import time
from collections import defaultdict
import os
from pyspark.sql import Row


# from pyspark.sql.functions import monotonically_increasing_id


def saveFile(*args, **kwargs):
    pass


def show(msg):
    date = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(date + "*****" + msg)


def spark_tools(data, keep_range=(0, 5000), save_var_list=None,
                tmp_file_name='cardnumber.txt', default_file_path='/chaofang/',
                map_list=None, good="0", bad="1", drop_keyword=None,
                enable_single_filter=True, single_threshold=0.8, include_none=True,
                response="code", enable_iv_filter=True,
                save_iv_dict=False, save_iv_file=None):
    """
    :param data: spark.datafame
    :param tmp_file_name: str;          default "cardnumber.txt";生成文件的临时文件名,存在这个文件即可
    :param default_file_path: str;      default '/chaofang/';datacomputer上生成的文件的路径(资源管理中)
    :param save_var_list: list          default None;传入变量名来存储指定的变量，变量不存在会以nan填充

    :param good: str;                   default "0";响应中的positive
    :param bad : str;                   default "1";响应中的negative
    :param map_list: list;              default ["id_value_stats", "mobile_value_stats"];字段中为map的list

    :param enable_single_filter: bool;  default True; 是否启用单变量筛选
    :param single_threshold: float;     default 0.8; 将保留小于该值的变量
    :param include_none: bool;          default True; 单一值计算时是否包含缺失值

    :param response: str;               default "code";响应列的列名
    :param enable_iv_filter: bool;      default True;是否启用iv筛选变量
    :param keep_range: tuple;           default (0,5000);选取iv前5000个变量

    :param save_iv_dict: bool;          default False:是否保存删完单一值之后所有变量的iv的dict
    :param save_iv_file: str;           default %Y-%m-%d-%H-%M-%S;保存iv的文件名

    example:
        def main(sparkSession):
            sql = "select * from creditaly.mogu_test_table"
            data = sparkSession.sql(sql)
            columns = spark_tools(data, default_file_path="/chaofang/",save_iv_file=True)
            print(columns)
    """
    start = time.time()
    map_list = ["id_value_stats", "mobile_value_stats"] if map_list is None else map_list
    if save_iv_file is None:
        save_iv_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def get_map_data(x, map_list, response='code'):
        """将map结构改为正常的row结构后供计算iv
        """
        tmp_dict = {}
        x = x.asDict()
        tmp_dict.update({response: x[response]})
        for map_name in map_list:
            pre = map_name[:1] + "_"
            tmp_dict.update({pre + k: v for k, v in x[map_name].iteritems()})
        return Row(**tmp_dict)

    def transformation_map2list(row, response="code"):
        '''各attr变量下不同value数量统计 for flatMap
        Return:由((attr,value,response),1) 组成的list
        '''
        row_dict = row.asDict()
        result = [((attr, row_dict.get(attr, None), row_dict.get(response, None)), 1) for attr in row_dict.keys()]
        return result

    def finally_get_cols(x, cols):
        x = x.asDict()
        for k, v in cols.iteritems():  # map_name, col_names
            pre = k[:1] + "_"
            if v is None:  # 取全部变量
                for k_, v_ in x[k].iteritems():
                    x.update({pre + k_: v_})
            else:  # 取指定的部分变量
                for i in v:
                    x.update({pre + i: x[k].get(i, np.nan)})
            del x[k]  # 删除这个map
        return Row(**x)

    def pre_cols_to_dict(attr, map_list):
        """将带前缀的变量名改为字典形式
        :param attr: list   带前缀的变量名
        :param map_list: list   map的列表
        :return: dict
        """
        cols_dict = defaultdict(list)
        map_dict = {i[:1]: i for i in map_list}
        for i in attr:
            tmp_ = i.split("_")
            cols_dict[map_dict[tmp_[0]]].append('_'.join(tmp_[1:]))
        return cols_dict

    def transformation_31to_13(x):
        """((name,value,label),count)->(name,(value,label,count))
        """
        key = x[0][0]
        value = [(x[0][1], x[0][2], x[1])]
        return (key, value)

    def calculation_iv(x, good="0", bad="1"):
        """计算iv
        """
        name = x[0]
        values = x[1]
        df_ = pd.DataFrame(values, columns=['value', 'label', 'count']).set_index(['label', 'value'])
        good_total = df_.xs(good)["count"].sum()
        bad_total = df_.xs(bad)["count"].sum()
        df_tmp = pd.concat([df_.xs(good), df_.xs(bad)], axis=1)
        df_tmp.columns = ["good_count", "bad_count"]
        df_tmp["inside_good_percent"] = df_tmp["good_count"].div(good_total)
        df_tmp["inside_bad_percent"] = df_tmp["bad_count"].div(bad_total)
        df_tmp["WOE"] = df_tmp["inside_good_percent"].div(
            df_tmp["inside_bad_percent"]).map(np.log)  # np.log做log计算,对series做对数计算
        df_tmp["IV"] = (df_tmp["inside_good_percent"].sub(
            df_tmp["inside_bad_percent"])).mul(df_tmp["WOE"])
        iv = df_tmp['IV'].sum()
        if iv is np.nan:
            iv = 0
        return (iv, name)

    def transformation_4to3(x):
        """((name,value,label),count)->((name,value),count))
        """
        key = (x[0][0], x[0][1])
        value = x[1]
        return (key, value)

    def drop_keyword_features(x, drop_keyword):
        tmp_dict = {}
        x = x.asDict()
        for k, v in x.iteritems():
            for key in drop_keyword:
                if key in k:
                    break  # 跳出这个循环
                else:
                    tmp_dict[k] = v
        return Row(**tmp_dict)

    show("开始计算数据")
    # data.cache()
    if save_var_list is not None:  # 提取特定的列的数据
        if isinstance(save_var_list, (list, tuple, set, np.ndarray)):
            show("检测到有save_var_list参数，自动关闭iv和single筛选")
            attr_list = save_var_list
            enable_iv_filter = False
            enable_single_filter = False
        else:
            raise ValueError("save_var_list值类型错误:{0}".format(type(save_var_list)))
    else:
        rdd = data.rdd.map(lambda x: get_map_data(x, map_list, response=response))
        if drop_keyword is not None:
            if not isinstance(drop_keyword, list):
                drop_keyword = [drop_keyword]
            show('Dropout features have "{0}"'.format(','.join(drop_keyword)))
            rdd = rdd.map(lambda x: drop_keyword_features(x, drop_keyword))
        attr_list = None
    if enable_iv_filter or enable_single_filter:
        show("开始计算数据分布")
        total = float(data.count())
        dist = rdd.flatMap(lambda x: transformation_map2list(x, response=response)).reduceByKey(add)
        dist.cache()
        show("样本数据总量为{0}".format(total))
        if enable_single_filter is True:  # 单一变量筛选
            show("开始进行单一变量筛选")
            tmp_rdd = dist.map(transformation_4to3).reduceByKey(add).filter(lambda x: x[1] / total > single_threshold)
            except_list = tmp_rdd.map(lambda x: x[0][0]).distinct().collect()
            show("删除变量数为:{0}".format(len(except_list)))
        if enable_iv_filter is True:  # iv筛选变量
            show("开始进行iv筛选")
            tmp_rdd = dist.map(transformation_31to_13).reduceByKey(add).map(lambda x: calculation_iv(x, good=good, bad=bad))
            if enable_single_filter:  # 排除单一性筛选剔除的元素
                ivs = tmp_rdd.filter(lambda x: x[1] not in except_list).sortByKey(ascending=False).collect()
            else:
                ivs = tmp_rdd.sortByKey(ascending=False).collect()
            ivs = ivs[keep_range[0]:keep_range[1]]
            attr_list = [iv[1] for iv in ivs]
    show("开始挑选筛选后的变量.")
    if attr_list is not None:
        cols = pre_cols_to_dict(attr_list, map_list)  # {map_name:[features]}
    else:  # 不进行单一性和iv筛选,取全部变量
        cols = zip(map_list, [None] * len(map_list))
    result = data.rdd.map(lambda x: finally_get_cols(x, cols))
    # result = result.toDF()
    show("耗时:{0}秒".format(int(time.time() - start)))
    return result


def write_to_local(result, filename, path=None, drop_tmp=True):
    """将结果先保存到hdfs上，然后在拉取到本地保存成csv文件.堡垒机环境下使用
    """
    start = time.time()
    show("开始保存变量...")
    result.write.csv(filename, mode='overwrite')
    header = result.columns
    local_tmp_path = filename + "_tmp.csv"  # 数据的临时文件名
    local_path = filename + ".csv"  # 最终的文件名
    if path is not None:
        local_tmp_path = os.path.join(path, local_tmp_path)
        local_path = os.path.join(path, local_path)
    os.system("hdfs dfs -getmerge {0} {1}".format(filename, local_tmp_path))  # 拉取数据
    df = pd.DataFrame([header])  # 头文件
    df.to_csv(local_path, header=False, index=False)
    show("合并列名和数据")
    os.system("cat {0} >> {1}".format(local_tmp_path, local_path))
    show("文件已保存到:{0}".format(local_path))
    show("耗时:{0}秒".format(int(time.time() - start)))
    if drop_tmp is True:
        show("开始删除中间文件...")
        os.system("hdfs dfs -rm -r {0}".format(filename))
        os.remove(local_tmp_path)


def get_csv_from_spark(data, filename, local_path=None,
                       enable_single_filter=True, single_threshold=0.8, include_none=True,
                       good="0", bad="1", response="code", keep_range=(0, 5000),
                       map_list=None, drop_tmp=True, enable_iv_filter=True):
    """原先堡垒机环境下函数
    """
    result = spark_tools(data, keep_range=keep_range,
                         good=good, bad=bad, enable_single_filter=enable_single_filter,
                         single_threshold=single_threshold, include_none=include_none,
                         response=response, map_list=map_list, enable_iv_filter=enable_iv_filter)
    result = result.toDF()
    write_to_local(result, filename=filename, path=local_path, drop_tmp=drop_tmp)


def dc_spark2table(data, tablename, spark,
                   enable_single_filter=True, single_threshold=0.8, include_none=True,
                   good="0", bad="1", response="code", keep_range=(0, 5000),
                   map_list=None, enable_iv_filter=True):
    """将数据挑选后写入dc表中
    """
    try:
        spark.sql("select * from creditaly.{0}".format(tablename))
    except Exception as e:
        show("ERROR:creditaly.{0}表不存在,请先创建数据库表".format(tablename))
        raise ValueError("{0}表不存在，请创建后使用".format(tablename))

    def to_map(x):
        r = {}
        x = x.asDict()
        r['select_features'] = x
        return Row(**r)

    result = spark_tools(data, keep_range=keep_range,
                         good=good, bad=bad, enable_single_filter=enable_single_filter,
                         single_threshold=single_threshold, include_none=include_none,
                         response=response, map_list=map_list, enable_iv_filter=enable_iv_filter)
    result = result.map(to_map).toDF()
    result.registerTempTable("tdl_rock_part1")
    show("开始写入{0}数据表中".format(tablename))
    sql = "insert overwrite table creditaly.{0} select select_features from tdl_rock_part1".format(tablename)
    spark.sql(sql)


if __name__ == '__main__':
    sql = "select * from creditaly.mogu_test_table"
    # data = spark.sql(sql)
    # data = spark.read.parquet('/user/var-compute/xiaotong.jiang/labels_mogu_170828')
    # result = spark_tools(data, keep_range=(0, 5000), response='code', save_iv_dict=False)
    # get_csv_from_spark(data, "mogu-test", keep_range=(0, 5000), response='code')
    # data = spark.read.parquet('/user/var-compute/dong.liu/jinshang_crdt_0531')
    # dc_spark2table(data, "mogu_part2_table", sparkSession, keep_range=(0, 5000), response='code')
    # data = spark.read.parquet('/user/var-compute/wenchao.yin/ch_fraud_171031')
    # response="fraud_tag"
