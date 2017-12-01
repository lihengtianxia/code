堡垒机环境使用指南
"$"开头的表示linux命令行下输入的内容,

解压压缩包得到3个文件
$ tar -xzvf rock_*****.tar.gz

spark_tools_v2.so
data_control_v2.so
st_functions.pyc
(原st步骤)进入spark环境并加载python包
参数说明:接口文档
$ /usr/install/spark2-yarn/bin/pyspark --py-files spark_tools_v2.so,st_functions.pyc
from spark_tools_v2 import get_csv_from_spark
data = spark.read.parquet('/user/var-compute/xiaotong.jiang/labels_mogu_170828')
get_csv_from_spark(data,"mogujie1027")
运行完之后会在本地生成一个文件mogujie1027.csv,用于后续步骤


(原dc步骤)计算分箱结果
参数说明:接口文档
from data_control_v2 import *
import pandas as pd
df_train = pd.read_csv("mogujie1027.csv")
df_test = pd.read_csv("mogu1013_test.csv") #测试集可以不填
dc=DcBins(df_train,'mogujie1027',df_test=df_test,drop_cols=['userid'],cut_method="cumsum",n_jobs=4，save_match_text)
#其他参数参考接口文档
dc.local_save_info() #保存原先info表
dc.local_save_m_t() #保存原先的M和T表


运行完后会在本地生成文件:
Info_mogujie1027.xlsx
T_Data_mogujie1027.csv
M_Data_mogujie1027.csv
(原mc步骤)计算评分卡
参数说明:接口文档
from model_control_v2 import *
mc=ModelControl('mogujie1027',exclude_column=['WOE_overdue_days_asC'],enable_train_test=True,add_info='code',save_proba=True)
mc.get_report()


dc平台环境使用指南
