以第二步的数据为基础进行评分卡计算
堡垒机环境接口文档
class ModelControl(object):
    def __init__(self, **kwargs):
        __doc__ =     """
必填
:param filename:str                 必填:文件名，不需要后缀,如mg1120,后台会自动去寻找对应的文件

交叉验证相关
:param enable_train_test:bool       default:False;True时读取T_Data...csv做交叉验证

变量筛选
:param filter_var:bool              default:True;是否进行变量筛选
:param response:str                 default:'code';相应列的列名
:param model_type:str               default:'sm';模型的类型,目前只支持sm
:param pvalue_limit:float           default:0.2;type为"sm"时生效，stepwise时的pvalue的limit
:param select_by:str                default:'iv';逐步回归会按select_by中的变量顺序来进行iv,ks,psi
:param enable_negative:bool         default:True;控制coef方向，False为正，None不做限制
:param exclude_column:list          default:None;建模时指定排除的column的内容，如['WOE_overdue_days_asC']

相关性筛选
:param enable_corr_filter:bool      default:True;是否进行相关性筛选
:param corr_limit:float             default:0.7;在启用相关性筛选时的筛选阈值

评分卡相关参数
:param bad                          default:1;
:param good                         default:0;
:param odds:float                   default:样本的坏好比;计算评分卡时的参数
:param base_score:int               default:580;基础分
:param double_score:int             default:50;翻倍分
:param round_score:bool             default:True;是否将评分四舍五入后输出
:param cut_method:'str'             default:'qcut';['qcut',['cut']
:param display_ks:int               default:10；计算ks时的分组数

输出相关
:param add_info:str or list         default:None;输出添加列，支持str和list
:param save_proba:bool              default:False;输出计算样本的分数和概率

备注:

手动指定基础数据文件参数(一般不用指定，调试的时候使用)
:param df_m:pandas.DataFrame        default:根据filename自动去获取;
:param df_t:pandas.DataFrame        default:根据filename自动去获取;
:param df_i:pandas.DataFrame        default:根据filename自动去获取;
:param df_i_d:pandas.DataFrame      default:根据filename自动去获取;
"""
