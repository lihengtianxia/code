从spark取数据并进行简单的筛选
堡垒机环境接口文档
def get_csv_from_spark(**kwargs):
    __doc__ = """
    堡垒机环境下使用的函数
    必填参数
    :param data: spark.datafame         必填;spark的dataframe
    :param filename: str                必填;需要保存的csv的名字,不需要后缀

    需要保存的文件参数
    :param local_path: str              default 当前路径;保存的csv的路径
    :param keep_range: tuple            default (0,5000);选取iv前5000个变量
    :param save_var_list: list                default None;传入变量名来存储指定的变量，变量不存在会以nan填充
    :param drop_tmp: bool               default True;是否删除中间文件

    响应列相关参数
    :param response: str;               default "code";响应列的列名
    :param good: str;                   default "0";响应中的positive
    :param bad : str;                   default "1";响应中的negative

    单一阈值相关参数
    :param enable_single_filter: bool;  default True; 是否启用单变量筛选
    :param single_threshold: float;     default 0.8; 将保留小于该值的变量
    :param include_none: bool;          default True; 单一值计算时是否包含缺失值

    IV筛选相关参数
    :param enable_iv_filter: bool;      default True;是否启用iv筛选变量
    :param map_list: list;              default ["id_value_stats", "mobile_value_stats"];字段中为map的list

    变量互匹字段批量过滤
    :param drop_keyword:list            default None;删除包含这些字段的特征
        如drop_keyword=["_15day","_30day"],会删除包含这两个字段的特征
        注意：如果填["60day"],会把包含"360day"的也删除掉，注意前面加"_"

    """
