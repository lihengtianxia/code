以第一步的数据为基础进行变量的分箱等操作
堡垒机环境接口文档
class DcBins(object):
    def __init__(self, **kwargs):
        __doc__ = """
        堡垒机环境下使用的函数
        必填:
        :param df_train: pandas.DataFrame           必填;需要计算的df
        :param filename: str                        保存的文件的主名字,不需要文件后缀名

        :param df_test: pandas.DataFrame            选填;测试集的df
        响应列相关参数
        :param drop_cols: list                      default:["uid", "umobile", "date"];忽略计算的列名
        :param bad: str                             default:1;响应列的坏标签
        :param good: str                            default:0;响应列的好标签
        :param response: str                        default:"code";响应列列名
        :param replace_options: dict                default:None;支持对不同列的变量的替换替换
            如对列"i_province2"中的"GS"替换为"CN";对列"i_test"中的-1111替换为-999
            {
            "i_province2":{"GS":"CN"},
            "i_test":{-1111:-999}
            }
            优先级大于replace_all字段
        :param replace_all:dict                     default:None;全局替换变量{key:value}用value替换key

        数值型分箱参数
        :param cut_method: str                      default:"cumsum";['quantile', 'cumsum','bestks']可选
        :param check_monotonicity: bool             default:True;是否检查单调性
        :param strict_monotonicity: bool            default:True;是否严格单调
        :param add_min_group: bool                  default:True;cumsum参数, 最小值表现充分时单独一组
        :param keep_separate_value: list            default:None;指定这些值需要单独一组(仅支持这组数据的最大or最小值)


        字符串型分箱参数
        :param single_group_num:int                 default:12;种类小于这个数特征，直接按值分箱
        :param max_of_length:int                    default:1000;种类大于这个数的特征，不进行分箱
        :param choice_by:str                        default:'iv';["iv","ks","woe","len"]其中一个，挑选分组时以什么为优先级考虑
        :param woe_inf_fill:str                     default:'avg';填充woe的方法，暂时只支持'avg'

        分箱共享参数
        :param fill_value: str                      default:"-999";空值的填充值
        :param max_cut_part: int                    default:10;最大分组数
        :param min_group_num: int                   default:3；最小分组数，包含空值列
        :param best_ks_loop: int                    default:3;best_ks分箱时的二分次数
        :param group_min_percent: float             default:0.05;每组的最小占比

        iv筛选参数
        :param enable_iv_threshold: bool            default:True;是否启用iv筛选
        :param iv_min: float                        default:0.02;保留iv大于iv_min的变量
        :param keep_range: int                      default:500;iv排序后保留的数量

        单一性筛选参数
        :param enable_single_threshold: bool        default:True;是否进行单一性筛选
        :param single_threshold: float              default:0.8;保留单一性小于single_threshold的变量
        :param include_none: bool                   default:True;计算单一性是否包含空值

        保存内容相关参数
        :param add_info: str or list                default:None;需要添加在M和T表后面的列，如['code','umobile']
        :param save_text: bool                      default:False;是否保持筛选后的区间数据
        :param save_match: bool                     default:False;是否保存筛选后匹配的woe数据
        :param save_match_text: bool                default:True;是否保存筛选后匹配的woe和区间
        :param save_tables_auto: bool               default:False;是否自动保存所有表,local_save_info和local_save_m_t手动保存

        多进程相关
        :param n_jobs: int                          default:4;进程数

        备注:
        choice_by是指在字符型分箱时，挑选choice_by指标最好的值.iv和最大，ks和最大，woe最大，len分组数最多
        keep_separate_value指定的值(如-1111)单独一组，则填写[-1111],只有在值是当前列最小或者最大的时候才生效
        replace_options和replace_all同时存在会先替换replace_options再替换replace_all
        当cut_method选择cumsum,并且add_min_group和keep_separate_value同时存在时，会优先使用add_min_group
        """

    def local_save_info():
        __doc__ = """
        保存原先的info表,filename为excel的名字，不需要加后缀.
        如："mogu1101",生成的文件名为:"info_mogu1101.xlsx"
        """

    def local_save_m_t(self, filename):
        __doc__ = """
        保存原先的M和T表,csv，不需要加后缀.
        如："mogu1101",生成的文件名为:"T_Data_mogu1101_***.csv","M_Data_mogu1101_***.csv"
        """
