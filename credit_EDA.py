import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import time

#查看数据集是否有重复行和方差为0的列
def duplicated_rows_and_Zero_var_cols(data):
    value_number=data.nunique()  #计算变量取值数量
    print('There are %s duplicated rows in training set'%data.duplicated().sum())
    print('There are %s Zero variance columns in training set'%len(value_number[value_number==1]))

#缺失值统计
def missing_values(data):
    data_na=pd.DataFrame(data.isnull().sum(), columns=['NAN_num'])
    data_na['NAN_rate'] = data.isnull().mean()
    data_na['dtype'] = data.dtypes 
    data_na = data_na[data_na['NAN_num']>0].sort_values(by='NAN_num',ascending=False)
    return data_na

#特征取值对预测目标影响的柱状图矩阵
def plot_bar_matrix(data,columns,target,row,col,x,y,w,h):
    fig=plt.figure(figsize=(x,y))
    sns.set(font_scale=2)
    for i,column in enumerate(columns):
        plt.subplot(row,col, i + 1)
        plt.subplots_adjust(wspace=w,hspace=h)
        data.groupby([column])[target].mean().plot.bar()
        plt.xticks(rotation=0)

#连续变量分布的直方图矩阵
def plot_hist_matrix(data,row,col,x,y,w,h):
    fig=plt.figure(figsize=(x,y))
    sns.set(font_scale=2)
    for i,column in enumerate(data):
        plt.subplot(row,col, i + 1)
        plt.subplots_adjust(wspace=w,hspace=h)
        data[column].plot.hist(bins=30)
        plt.xlabel(column)

#计算特征和预测目标的互信息
def mutual_info(data,target):
    mi,index=[],[]
    for column in data.columns:
        index.append(column)
        if data[column].dtypes=='object':
            mi.append(normalized_mutual_info_score(data[target],data[column].astype(str)))
        else:
            mi.append(normalized_mutual_info_score(data[target],data[column]))
    data_mi=pd.Series(data=mi,index=index).sort_values(ascending=False)
    return data_mi

#基于相关系数找出冗余特征
def find_redundancy(df,correlation):
    cor=df.corr()
    cor_columns=cor.columns
    cor_array=np.array(cor)
    cor_tuple=np.where(abs(cor_array)>correlation)
    list_1=list(cor_tuple[0])
    list_2=list(cor_tuple[1])
    n,redundancy=0,[]
    for i,x in enumerate(list_1):
        if x < list_2[i]:
            n+=1
            redundancy.append(cor_columns[list_2[i]])
            print(n,'\t',cor_columns[x],' & ',cor_columns[list_2[i]])
    print('There are {0} redundant variables in {1} variables'.format(len(set(redundancy)),len(cor.columns)))

#基于相关系数删除冗余特征
def delete_redundancy(df,correlation):
    cor=df.corr()
    cor_columns=cor.columns
    cor_array=np.array(cor)
    cor_tuple=np.where(abs(cor_array)>correlation)
    list_1=list(cor_tuple[0])
    list_2=list(cor_tuple[1])
    n,redundancy=0,[]
    for i,x in enumerate(list_1):
        if x < list_2[i]:
            n+=1
            redundancy.append(cor_columns[list_2[i]])
    redundancy=list(set(redundancy))
    print('%s redundant variables are deleted'%len(redundancy))
    return df[[x for x in df.columns if x not in redundancy]]

#独热编码
def one_hot_encoder(df):
    columns=df.dtypes[df.dtypes=='object'].index
    one_hot=pd.get_dummies(df[columns],dummy_na=True)
    new_df=pd.concat([df,one_hot],axis=1)
    new_df.drop(columns,axis=1,inplace=True)
    return new_df,list(one_hot)

#下采样
def undersampling(data,target):
    abnormal_number=len(data[data[target]==1])
    abnormal_index=data[data[target]==1].index.tolist()
    normal_index=data[data[target]==0].index
    normal_undersampling=np.random.choice(normal_index,abnormal_number,replace=False).tolist()
    undersampling_index=abnormal_index
    undersampling_index.extend(normal_undersampling)
    new_data=data.loc[undersampling_index,:]
    return new_data

#lightGBM对比下采样和不采用的AUC和计算耗时
def compare_undersampling_and_full(df,df_under):
    x_full,y_full=df.iloc[:,1:],df.iloc[:,0]
    x_under,y_under=df_under.iloc[:,1:],df_under.iloc[:,0]
    lgb=LGBMClassifier(objective='binary',learning_rate=0.05,max_depth=5,
                   num_leaves=100,max_bin=200)
    time1=time.perf_counter()
    auc_score1=cross_val_score(lgb,x_full,y_full,cv=3,scoring='roc_auc').mean()
    time2=time.perf_counter()
    auc_score2=cross_val_score(lgb,x_under,y_under,cv=3,scoring='roc_auc').mean()
    time3=time.perf_counter()
    print('full_data:auc={},computing_time={}seconds'.format(round(auc_score1,4),round(time2-time1,2)))
    print('undersampling:auc={},computing_time={}seconds'.format(round(auc_score2,4),round(time3-time2,2)))

#一、分析主要的测试集和训练集数据
#1.1查看数据概况
train=pd.read_csv('D:/data_project/credit/application_train.csv')
test=pd.read_csv('D:/data_project/credit/application_test.csv')
train.columns=train.columns.map(str.lower)
test.columns=test.columns.map(str.lower)
pd.set_option("display.max_columns",200)
pd.set_option("display.max_rows",100)
train.info()
print('-'*40)
test.info()
train.head()


#1.2查看训练集是否有重复行和方差为0的列
#duplicated_rows_and_Zero_var_cols(train)


#1.3查看正负样本比例
train.target.value_counts(normalize=True)


#1.4查看总体数据缺失情况
total=pd.concat([train,test],ignore_index=True,sort=False)
missing=missing_values(total)
print(missing)

#1.5 object类别变量分析
#查看所有类别变量的类别数量
total.select_dtypes('object').nunique()

#查看数据异常code_gender,删除异常值
print(train.code_gender.value_counts())
print('-'*40)
print(test.code_gender.value_counts())
total=total[total.code_gender != 'XNA']

#查看存在缺失值的object变量情况
miss_object=missing[missing.dtype=='object'].index
for col in miss_object: print(total[col].value_counts())

#查看所有类别变量的取值和违约率的关系
category_columns=total.dtypes[total.dtypes=='object'].index
train=total[total.target.notnull()]
train[category_columns]=train[category_columns].fillna('na')
#plot_bar_matrix(train,category_columns,'target',4,4,30,25,0.2,0.25)


#1.6区分数值类型中的类别变量和连续变量
#查看所有int&flaot变量的取值数量,并判断是否异常
print(total.select_dtypes('int64').nunique())
print('-'*40)
print(total.select_dtypes('float64').nunique())

#查看region_rating_client_w_city变量的异常取值,推断可能的取值替代异常
print(total.region_rating_client_w_city.value_counts())
print(total[total.region_rating_client_w_city==-1])
mode=total[total.region_rating_client==2].region_rating_client_w_city.mode()[0].astype(np.int64)
total['region_rating_client_w_city'].replace(-1,mode,inplace=True)

#提取数值类型中的分类变量和连续变量的列名
int_cols=total.dtypes[total.dtypes=='int64'].index.tolist()
float_cols=total.dtypes[total.dtypes=='float64'].index.tolist()
numeric_category=int_cols[5:]
numeric_continuous=[x for x in float_cols+int_cols[1:5] if x !='target']


#1.7分析数值型分类变量
#变量的取值对目标的影响
#plot_bar_matrix(train,numeric_category,'target',7,5,30,30,0.2,0.3)

#查看二值变量取值
num=total[numeric_category].nunique()
two_values=num[num==2].index
total[two_values].apply(pd.Series.value_counts)


#1.8查看数值连续变量的分布和异常情况
#描述性统计
total[numeric_continuous].describe()

#amt_income_total离群点,对比训练集和测试上的数据分布
print(train.amt_income_total.describe())
print(test.amt_income_total.describe())
print(train[train.amt_income_total>1e7].shape[0])

#days_employed异常,用缺失值代替
print(train[train.days_employed==365243].target.mean())
print(train[train.days_employed!=365243].target.mean())
total['days_employed'].replace(365243,np.nan,inplace=True)

#直方图矩阵,观察所有连续变量的数据分布
days_columns=[x for x in numeric_continuous if 'days' in x]
total[days_columns]=total[days_columns]/-365  #days变量转换为年
plot_hist_matrix(total[numeric_continuous],12,6,40,55,0.2,0.3)

#own_car_age异常，缺失值替代
total.loc[total.own_car_age>60,'own_car_age']=np.nan

#查看own_car_age和flag_own_car变量取值矛盾的的异常情况
print(total[total.own_car_age.isnull()][total.flag_own_car=='Y'].shape[0])
print(total[total.own_car_age.notnull()][total.flag_own_car=='N'].shape[0])


#1.9创建新变量
total['income/family']=total['amt_income_total']/total['cnt_fam_members']
total['income/credit']=total['amt_income_total']/total['amt_credit']
total['income/annuity']=total['amt_income_total']/total['amt_annuity']
total['annuity/credit']=total['amt_annuity']/total['amt_credit']
total['employed/birth']=total['days_employed']/total['days_birth']
#查看是否有异常
new_cols=['income/family','income/credit','income/annuity','annuity/credit','employed/birth']
total[new_cols].describe()

#1.10 数值变量相关性分析
#person相关系数，找出相关度大于0.9的变量
'''find_redundancy(total,0.9)
#散点图查看几组相关变量
fig=plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.scatter(total.amt_goods_price,total.amt_credit)
plt.subplot(1,2,2)
plt.scatter(total.apartments_avg,total.apartments_medi)'''
#删除相关度大于0.9的冗余变量
total=delete_redundancy(total,0.9)


#1.11 数值型连续变量离散化，查看变量取值对目标的影响
#自定义区间离散化
train=total[total.target.notnull()]
train['cnt_fam_members']=pd.cut(train.cnt_fam_members,[0,1,2,3,4,30]).astype('str')
train['cnt_children']=pd.cut(train.cnt_children,[-1,0,1,2,3,20]).astype('str')
train['obs_30_cnt_social_circle']=pd.cut(train.obs_30_cnt_social_circle,[-1,1,6,11,500],labels=list(range(4))).astype('str')
train['def_30_cnt_social_circle']=pd.cut(train.def_30_cnt_social_circle,[-1,0,1,2,3,50],labels=list(range(5))).astype('str')
train['def_60_cnt_social_circle']=pd.cut(train.def_60_cnt_social_circle,[-1,0,1,2,3,50],labels=list(range(5))).astype('str')
train['amt_req_credit_bureau_hour']=train.amt_req_credit_bureau_hour.map(lambda x : 1 if x>0 else x).astype('str')
train['amt_req_credit_bureau_day']=train.amt_req_credit_bureau_day.map(lambda x : 1 if x>0 else x).astype('str')
train['amt_req_credit_bureau_week']=train.amt_req_credit_bureau_week.map(lambda x : 1 if x>0 else x).astype('str')
train['amt_req_credit_bureau_mon']=pd.cut(train.amt_req_credit_bureau_mon,[-1,0,1,500],labels=list(range(3))).astype('str')
train['amt_req_credit_bureau_qrt']=pd.cut(train.amt_req_credit_bureau_qrt,[-1,0,1,500],labels=list(range(3))).astype('str')
train['amt_req_credit_bureau_year']=pd.cut(train.amt_req_credit_bureau_year,[-1,1,3,500],labels=list(range(3))).astype('str')
train['elevators_avg']=train.elevators_avg.map(lambda x : 1 if x>0 else x).astype('str')
train['nonlivingapartments_avg']=train.nonlivingapartments_avg.map(lambda x : 1 if x>0 else x).astype('str')
train['nonlivingarea_avg']=train.nonlivingarea_avg.map(lambda x : 1 if x>0 else x).astype('str')
train['floorsmax_avg']=pd.cut(train.floorsmax_avg,[-1,0.15,0.2,0.35,1]).astype('str')
train['floorsmin_avg']=pd.cut(train.floorsmin_avg,[-1,0.1,0.25,0.4,1]).astype('str')
#其余数值变量，等频离散化
other_float=train.dtypes[train.dtypes=='float64'].index.tolist()
other_numeric=[x for x in other_float+int_cols[2:5] if x !='target']
a=[]
for column in other_numeric:
    try:
        train[column]=pd.qcut(train[column],5,list(range(5))).astype('str')
    except:
        a.append(column)
#柱状图，变量和目标之间的关系
numeric_continuous=[x for x in numeric_continuous if x in list(train)]
plot_bar_matrix(train,numeric_continuous,'target',6,6,40,35,0.2,0.3)


#二、提取外部信息数据
#2.1提取征信局数据
#查看数据概况
bureau=pd.read_csv('D:/data_project/credit/bureau.csv')
bb=pd.read_csv('D:/data_project/credit/bureau_balance.csv')
bureau.info()
print('-'*40)
bb.info()
#描述性统计,查看是否有异常值
bureau.describe(include='all')
bb.describe(include='all')
#独热化
bb,bb_cols=one_hot_encoder(bb)
bureau,bureau_cols=one_hot_encoder(bureau)
#bureau_balance数据聚合
bb_aggregations={'MONTHS_BALANCE':['min','max','size']}
for col in bb_cols:
    bb_aggregations[col]='mean'
bb_agg=bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
bb_agg.columns=[x[0]+'_'+x[1] for x in list(bb_agg)]  #列名修改
#合并数据集
bureau=pd.merge(bureau,bb_agg,left_on='SK_ID_BUREAU',right_index=True,how='left')
bureau.drop(['SK_ID_BUREAU'],axis=1,inplace=True)
#bureau数据聚合
num_cols=list(bureau)[1:list(bureau).index('AMT_ANNUITY')+1]
bureau_aggregations = {'MONTHS_BALANCE_min': 'min',
                       'MONTHS_BALANCE_max': 'max',
                       'MONTHS_BALANCE_size': ['mean', 'sum']}
for col in num_cols:
    bureau_aggregations[col]=['min','max','mean','sum','var']
for col in bureau_cols:
    bureau_aggregations[col]='mean'
for col in bb_cols:
    bureau_aggregations[col+'_mean']='mean'
bureau_agg=bureau.groupby('SK_ID_CURR').agg(bureau_aggregations)
bureau_agg.columns=[x[0]+'_'+x[1] for x in list(bureau_agg)]  #列名修改
bureau_agg['bureau_count']=bureau.groupby('SK_ID_CURR').size()   #提取记录次数
#删除冗余变量
bureau_agg=delete_redundancy(bureau_agg,0.9)
del bureau,bb,bb_agg


#2.2、提取信用卡消费数据
#查看概况
credit_card=pd.read_csv('D:/data_project/credit/credit_card_balance.csv')
credit_card.info()
credit_card.describe()
#数据处理
credit_card,object_cols=one_hot_encoder(credit_card)#独热化
credit_card.drop('SK_ID_PREV',axis=1,inplace=True)#删除'SK_ID_PREV'
#聚合
num_cols=list(credit_card)[1:list(credit_card).index('SK_DPD_DEF')+1]
cc_aggregations={}
for col in num_cols:
    cc_aggregations[col]=['min','max','mean','sum','var']
for col in object_cols:
    cc_aggregations[col]='mean'
cc_agg=credit_card.groupby('SK_ID_CURR').agg(cc_aggregations)
cc_agg.columns=[x[0]+'_'+x[1] for x in list(cc_agg)]  #列名修改
cc_agg['cc_count']=credit_card.groupby('SK_ID_CURR').size() #提取记录次数
#删除冗余变量
cc_agg=delete_redundancy(cc_agg,0.9)
del credit_card


#2.3、提取历史贷款数据
#查看概况
pre_apply=pd.read_csv('D:/data_project/credit/previous_application.csv')
pre_apply.info()
pre_apply.describe()
#异常值处理
abnormal_cols=list(pre_apply)[-6:-1]
for col in abnormal_cols:
    pre_apply[col].replace(365243,np.nan,inplace=True)
#数据处理
pre_apply,object_cols=one_hot_encoder(pre_apply)#独热化
pre_apply.drop('SK_ID_PREV',axis=1,inplace=True)#删除'SK_ID_PREV'
#聚合
num_cols=list(pre_apply)[1:list(pre_apply).index('NFLAG_INSURED_ON_APPROVAL')+1]
pre_aggregations={}
for col in num_cols:
    pre_aggregations[col]=['min','max','mean','sum','var']
for col in object_cols:
    pre_aggregations[col]='mean'
pre_agg=pre_apply.groupby('SK_ID_CURR').agg(pre_aggregations)
pre_agg.columns=[x[0]+'_'+x[1] for x in list(pre_agg)]  #列名修改
pre_agg['pre_count']=pre_apply.groupby('SK_ID_CURR').size()   #提取记录次数
#删除冗余变量
pre_agg=delete_redundancy(pre_agg,0.9)
del pre_apply


#2.4、提取pos_cash数据
#查看概况
pos=pd.read_csv('D:/data_project/credit/POS_CASH_balance.csv')
pos.info()
pos.describe()
pos.select_dtypes('object').nunique()
#数据处理
pos,object_cols=one_hot_encoder(pos)#独热化
pos.drop('SK_ID_PREV',axis=1,inplace=True)#删除'SK_ID_PREV'
#聚合
num_cols=list(pos)[1:list(pos).index('SK_DPD_DEF')+1]
pos_aggregations={}
for col in num_cols:
    pos_aggregations[col]=['min','max','mean','sum','var']
for col in object_cols:
    pos_aggregations[col]='mean'
pos_agg=pos.groupby('SK_ID_CURR').agg(pos_aggregations)
pos_agg.columns=[x[0]+'_'+x[1] for x in list(pos_agg)]  #列名修改
pos_agg['pos_count']=pos.groupby('SK_ID_CURR').size()   #提取记录次数
#基于相关性删除冗余变量
pos_agg=delete_redundancy(pos_agg,0.9)
del pos


#2.5、提取payments数据
#读入数据，查看概况
pay=pd.read_csv('D:/data_project/credit/installments_payments.csv')
pay.info()
pay.drop(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'],axis=1,inplace=True)
pay.describe()
#创建分期支付率变量
pay['pay_rate']=pay['AMT_PAYMENT']/pay['AMT_INSTALMENT']
pay['pay_diff']=pay['DAYS_ENTRY_PAYMENT']-pay['DAYS_INSTALMENT']
#查看数据分布，处理异常值
pay[['pay_rate','pay_diff']].describe()
pay['pay_rate']=pay['pay_rate'].map(lambda x:1 if x>1 else x)
#聚合
pay_aggregations={'NUM_INSTALMENT_VERSION':'nunique'}
for col in list(pay)[2:]:
    pay_aggregations[col]=['min','max','mean','sum','var']
pay_agg=pay.groupby('SK_ID_CURR').agg(pay_aggregations)
pay_agg.columns=[x[0]+'_'+x[1] for x in list(pay_agg)]  #列名修改
pay_agg['pay_count']=pay.groupby('SK_ID_CURR').size()   #提取记录次数
#基于相关性删除冗余变量
pay_agg=delete_redundancy(pay_agg,0.9)
del pay

#三、合并数据后分析特征对信贷违约风险的影响
#3.1合并数据
for new_df in [bureau_agg,pre_agg,pay_agg,pos_agg,cc_agg]:
    total=pd.merge(total,new_df,left_on='sk_id_curr',right_on='SK_ID_CURR',how='left',sort=False)
total.drop('sk_id_curr',axis=1,inplace=True)
del bureau_agg,pre_agg,pay_agg,pos_agg,cc_agg

#3.2数据概况
total.info()

#3.3查看数据缺失情况,
na_rate=missing_values(total)
print(na_rate)
#删除缺失率超过90%的变量
high_na=na_rate[na_rate.NAN_rate>0.9].index.tolist()
total.drop(high_na,axis=1,inplace=True)

#3.4查看是否有方差为0的列
value_number=total.nunique()
print(len(value_number[value_number==1]))
#查看唯一值
cols=value_number[value_number==1].index.tolist()
total[cols].apply(pd.Series.unique)
#填补方差为0变量的缺失值
value_number=total.nunique()
cols=value_number[value_number==1].index.tolist()
total[cols]=total[cols].fillna(1)

#3.6基于互信息分析单变量的特征重要度
train=total[total.target.notnull()]
mi=mutual_info(train,'target')
print(mi[1:30])     #最相关的变量
print(mi[-100:])    #最不相关的变量
#删除低互信息特征
total=total[mi[mi>0.0001].index]

#total.to_csv("d:/data_project/credit/data1.csv", index = False)
total=pd.read_csv('D:/data_project/credit/data1.csv')


#3.8对比下采样和不采样的auc值
object_cols=total.dtypes[total.dtypes=='object'].index
total[object_cols]=total[object_cols].astype('category')
train=total[total.target.notnull()]
train_under=undersampling(train,'target')#下采样
compare_undersampling_and_full(train,train_under)





