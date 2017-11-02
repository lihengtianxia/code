# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
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


def spark_tools(data, keep_range=(0, 5000),
                tmp_file_name='cardnumber.txt', default_file_path='/chaofang/',
                map_list=None, good="0", bad="1",
                enable_single_filter=True, single_threshold=0.8, include_none=True,
                response="code", enable_iv_filter=True,
                save_iv_dict=False, save_iv_file=None):
    """
    :param data: spark.datafame
    :param tmp_file_name: str;          default "cardnumber.txt";生成文件的临时文件名,存在这个文件即可
    :param default_file_path: str;      default '/chaofang/';datacomputer上生成的文件的路径(资源管理中)

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
        tmp_dict.update({'code': x[response]})
        for map_name in map_list:
            pre = map_name[:1] + "_"
            for k, v in x[map_name].iteritems():
                tmp_dict[pre + k] = v
        return Row(**tmp_dict)

    def attr_cnt(row, attr_list):
        '''各attr变量下不同value数量统计 for flatMap
        Return:由((attr,value),1) 组成的list
        '''
        row_dict = row.asDict()
        result = []
        for attr in attr_list:
            result.append(((attr, row_dict.get(attr, None)), 1))
        return result

    def finally_get_cols(x, cols):
        x = x.asDict()
        for k, v in cols.iteritems():  # map_name, col_names
            pre = k[:1] + "_"
            for i in v:
                x.update({pre + i: x[k][i]})
            del x[k]
        return Row(**x)

    def attr_to_dict(attr, map_list):
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

    print("开始计算数据...")
    rdd = data.rdd.map(lambda x: get_map_data(x, map_list, response=response))
    attr_list = list(set(rdd.toDF().columns))
    print("总变量数为:{0}".format(len(attr_list)))
    if enable_iv_filter or enable_single_filter:
        data_good = rdd.filter(lambda x: x["code"] == good)
        data_bad = rdd.filter(lambda x: x["code"] == bad)
        print("开始计算数据分布...")
        dist_good = data_good.flatMap(lambda x: attr_cnt(x, attr_list)).reduceByKey(add).collect()
        dist_bad = data_bad.flatMap(lambda x: attr_cnt(x, attr_list)).reduceByKey(add).collect()

        df_good = pd.DataFrame([(attr, val, count) for ((attr, val), count) in dist_good], columns=['attr', 'value', 'good_count']).set_index(['attr', 'value'])
        df_bad = pd.DataFrame([(attr, val, count) for ((attr, val), count) in dist_bad], columns=['attr', 'value', 'bad_count']).set_index(['attr', 'value'])
        df = pd.concat([df_bad, df_good], axis=1)
        if enable_single_filter is True:  # 单一变量筛选
            print("开始进行单一变量筛选...")
            result_list = []
            good_total = df.xs(attr_list[0])["good_count"].sum()
            bad_total = df.xs(attr_list[0])["bad_count"].sum()
            total = good_total + bad_total
            for _ in attr_list:
                df_tmp = df.xs(_)
                if include_none is False:  # 是否包含空值
                    try:
                        df_tmp = df_tmp.drop(np.nan, axis=0)
                    except ValueError as e:
                        pass
                df_tmp["percent"] = (df_tmp["good_count"] + df_tmp["bad_count"]) / total
                if df_tmp["percent"].max() <= single_threshold:
                    result_list.append(_)
            attr_list = result_list
            print("剩余变量数为:{0}".format(len(result_list)))

        if enable_iv_filter is True:  # iv筛选变量
            print("开始进行iv筛选...")
            good_total = df.xs(attr_list[0])["good_count"].sum()
            bad_total = df.xs(attr_list[0])["bad_count"].sum()
            ivs = dict()
            for _ in attr_list:
                df_tmp = df.xs(_)
                df_tmp["inside_good_percent"] = df_tmp["good_count"].div(good_total)
                df_tmp["inside_bad_percent"] = df_tmp["bad_count"].div(bad_total)
                df_tmp["WOE"] = df_tmp["inside_good_percent"].div(
                    df_tmp["inside_bad_percent"]).map(np.log)  # np.log做log计算,对series做对数计算
                df_tmp["IV"] = (df_tmp["inside_good_percent"].sub(
                    df_tmp["inside_bad_percent"])).mul(df_tmp["WOE"])
                iv = df_tmp['IV'].sum()
                ivs[_] = iv
            ivs = sorted(ivs.iteritems(), key=lambda k: k[1], reverse=True)
            if save_iv_dict is True:  # 保存iv结果到文件
                f = open(tmp_file_name, "wb")
                f.write(json.dumps(dict(ivs), indent=4))
                filename = default_file_path + save_iv_file + ".json"
                saveFile(filename, f)
                print("iv数据已保存到:{0}".format(filename))

            ivs = ivs[keep_range[0]:keep_range[1]]
            attr_list = [iv[0] for iv in ivs]
    # 抽取符合条件的列
    print("开始保存筛选后的变量...")
    cols = attr_to_dict(attr_list, map_list)
    result = data.rdd.map(lambda x: finally_get_cols(x, cols))
    print("耗时:{0}秒".format(int(time.time() - start)))
    return result.toDF()


def write_to_local(result, filename, local_path=None, drop_tmp=True):
    result.write.csv("mogu-test", header=True)
    if local_path is None:
        local_tmp_path = filename + "_tmp.csv"
        local_path = filename + ".csv"
    else:
        local_tmp_path = os.path.join(local_path, filename + "_tmp.csv")
        local_path = os.path.join(local_path, filename + ".csv")
    os.system("hdfs dfs -getmerge {0} {1}".format(filename, local_tmp_path))
    df = pd.read_csv(local_tmp_path)
    df = df[df.ix[:, 0] != df.columns[0]]
    df.to_csv(local_path, header=True, index=False)
    print("文件已保存到:{0}".format(local_path))
    if drop_tmp is True:
        print("开始删除中间文件...")
        os.system("hdfs dfs -rm -r {0}".format(filename))
        os.remove(local_tmp_path)


def get_csv_from_spark(data, filename, local_path=None,
                       enable_single_filter=True, single_threshold=0.8, include_none=True,
                       good="0", bad="1", response="code", keep_range=(0, 5000),
                       map_list=None, drop_tmp=True):
    result = spark_tools(data, keep_range=keep_range,
                         good=good, bad=bad, enable_single_filter=enable_single_filter,
                         single_threshold=single_threshold, include_none=include_none,
                         response=response, map_list=map_list)
    write_to_local(result, filename=filename, local_path=local_path, drop_tmp=drop_tmp)


# if __name__ == '__main__':
#     # sql = "select * from creditaly.mogu_test_table2"
#     # data = spark.sql(sql)
#     data = spark.read.parquet('/user/var-compute/xiaotong.jiang/labels_mogu_170828')
#     # result = spark_tools(data, keep_range=(0, 50), response='code', save_iv_dict=False)
#     get_csv_from_spark(data, "mogu-test", keep_range=(0, 50), response='code')
