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


def log(msg):
    date = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(date + "*****" + msg)


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
        result = [((attr, row_dict.get(attr, None)), 1) for attr in attr_list]
        # for attr in attr_list:
        #     result.append(((attr, row_dict.get(attr, None)), 1))
        return result

    def attr_cnt2(row, attr_list, response="code"):
        '''各attr变量下不同value数量统计 for flatMap
        Return:由((attr,value),1) 组成的list
        '''
        row_dict = row.asDict()
        result = [((attr, row_dict.get(attr, None), row_dict.get(response, None)), 1) for attr in attr_list]
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

    def transformation(x):
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

    log("开始计算数据")
    rdd = data.rdd.map(lambda x: get_map_data(x, map_list, response=response))
    rdd.cache()
    attr_list = list(set(rdd.toDF().columns))
    log("总变量数为:{0}".format(len(attr_list)))
    if enable_iv_filter or enable_single_filter:
        log("开始计算数据分布")
        total = float(rdd.count())
        if enable_single_filter is True:  # 单一变量筛选
            log("开始进行单一变量筛选")
            rdd2 = rdd.flatMap(lambda x: attr_cnt(x, attr_list)).reduceByKey(add)
            rdd3 = rdd2.filter(lambda x: x[1] / total > single_threshold)
            except_cols = set(rdd3.map(lambda x: x[0][0]).collect())
            attr_list = list(set(attr_list) - except_cols)
            log("剔除变量数为:{0}".format(len(except_cols)))
            log("剩余变量数为:{0}".format(len(attr_list)))
        if enable_iv_filter is True:  # iv筛选变量
            log("开始进行iv筛选")
            dist = rdd.flatMap(lambda x: attr_cnt2(x, attr_list, response=response)).reduceByKey(add)
            ivs = dist.map(transformation).reduceByKey(add).map(calculation_iv).sortByKey(ascending=False).collect()
            if save_iv_dict is True:  # 保存iv结果到文件
                f = open(tmp_file_name, "wb")
                f.write(json.dumps({v: i for i, v in ivs}, indent=4))
                filename = default_file_path + save_iv_file + ".json"
                saveFile(filename, f)
                log("iv数据已保存到:{0}".format(filename))
            ivs = ivs[keep_range[0]:keep_range[1]]
            attr_list = [iv[1] for iv in ivs]
    log("开始挑选筛选后的变量.")
    cols = attr_to_dict(attr_list, map_list)
    result = data.rdd.map(lambda x: finally_get_cols(x, cols))
    # result = result.toDF()
    log("耗时:{0}秒".format(int(time.time() - start)))
    return result


def write_to_local(result, filename, local_path=None, drop_tmp=True):
    """将结果先保存到hdfs上，然后在拉取到本地保存成csv文件.堡垒机环境下使用
    """
    start = time.time()
    log("开始保存变量...")
    result.write.csv(filename, mode='overwrite')
    header = result.columns
    if local_path is None:
        local_tmp_path = filename + "_tmp.csv"
        local_path = filename + ".csv"
    else:
        local_tmp_path = os.path.join(local_path, filename + "_tmp.csv")
        local_path = os.path.join(local_path, filename + ".csv")
    os.system("hdfs dfs -getmerge {0} {1}".format(filename, local_tmp_path))
    log("合并列名和数据")
    df = pd.read_csv(local_tmp_path, names=header)  # todo:有优化空间
    # df = df[df.ix[:, 0] != df.columns[0]]
    df.to_csv(local_path, header=True, index=False)
    log("文件已保存到:{0}".format(local_path))
    log("耗时:{0}秒".format(int(time.time() - start)))
    if drop_tmp is True:
        log("开始删除中间文件...")
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
    write_to_local(result, filename=filename, local_path=local_path, drop_tmp=drop_tmp)


def dc_spark2table(data, tablename, spark,
                   enable_single_filter=True, single_threshold=0.8, include_none=True,
                   good="0", bad="1", response="code", keep_range=(0, 5000),
                   map_list=None, enable_iv_filter=True):
    """将数据挑选后写入dc表中
    """
    try:
        spark.sql("select * from creditaly.{0}".format(tablename))
    except Exception as e:
        log("ERROR:creditaly.{0}表不存在,请先创建数据库表".format(tablename))
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
    log("开始写入{0}数据表中".format(tablename))
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
