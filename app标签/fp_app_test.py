
'''
匹配APP回溯数据操作指南 V0.1

* APP相关变量可以按月回溯。回溯匹配逻辑为按照贷款申请时间点回溯匹配历史APP标签数据库，选择申请时间点所在月份前最新一次的指标标签，时间分区字段为ds。

'''


'''3.1 匹配手机维度APP相关变量'''

# test为测试数据pyspark dataframe，mobile字段为测试数据的主键手机号,date为申请日期字段,默认取前六位代表年月(Type:String; Eg: '201706')以匹配app变量的时间分区ds(Type:String)

fp_df = spark.sql('''   SELECT *
                        FROM
                        (SELECT *,
                        row_number() OVER(PARTITION BY t.accountmobile ORDER BY t.ds DESC) rank
                        FROM
                            (SELECT *
                            FROM test inner join prodmodel.fp_mobile_mt as t
                            on test.mobile = t.accountmobile
                            WHERE substring(test.date,1,6) >= t.ds
                            )
                        ) tmp
                        WHERE rank = 1''').drop("rank")












'''3.2 匹配身份证维度APP相关变量'''

# test为测试数据pyspark dataframe，id字段为测试数据的主键身份证号,date为申请日期字段,默认取前六位代表年月(Type:String; Eg: '201706')以匹配app变量的时间分区ds(Type:String)


fp_df = spark.sql('''   SELECT *
                        FROM
                        (SELECT *,
                        row_number() OVER(PARTITION BY t.idnumber ORDER BY t.ds DESC) rank
                        FROM
                            (SELECT *
                            FROM test inner join fp_id_mt as t
                            on test.id = t.idnumber
                            WHERE substring(test.date,1,6) >= t.ds
                            )
                        ) tmp
                        WHERE rank = 1''').drop("rank")


