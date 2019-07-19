import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score,learning_curve,RandomizedSearchCV,StratifiedKFold,GridSearchCV
from sklearn.cluster import KMeans

#一、定义函数
#1.1单一功能的函数
#缺失值统计
def missing_values(data):
    data_na=pd.DataFrame(data.isnull().sum(), columns=['NAN_num'])
    data_na['NAN_rate'] = data.isnull().mean()
    data_na['dtype'] = data.dtypes 
    data_na = data_na[data_na['NAN_num']>0].sort_values(by='NAN_num',ascending=False)
    return data_na

#根据相关类别对应的众数填补类别缺失
def fill_class_by_mode(data,column_null,column_notnull,fill_data):
    for i in fill_data.index:
        v = fill_data.loc[i,column_notnull]
        mode=data[data[column_notnull]==v][column_null].mode()[0]
        data.loc[i,column_null]=mode

#回归填补缺失值,使用相关的前k个不含缺失值的变量
def regression_fill(data,cor,column,clf,k):
    all_columns=abs(cor[column]).nlargest(k).index.tolist()
    notnull_columns=[x for x in all_columns if data[x].isnull().sum()==0 or x==column]
    data_notnull=data.loc[(data[column].notnull()),notnull_columns]
    data_isnull=data.loc[(data [column].isnull()),notnull_columns]
    x_train=data_notnull.drop([column],axis=1)
    y_train=data_notnull[column]
    x_test=data_isnull.drop([column],axis=1)
    clf.fit(x_train,y_train)
    data_predict=clf.predict(x_test)
    data.loc[data[column].isnull(),[column]]=data_predict

#计算变量和预测目标的互信息系数
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

#连续特征离散化,等频+聚类分组
def discretize_variate(df,cols):
    df[cols]=StandardScaler().fit_transform(df[cols])
    for col in cols:
        if df[col].nunique()<=5:
            df[col]=df[col].astype('str')
        else:
            try:
                df[col]=pd.qcut(df[col],5,list(range(5))).astype('str')
            except:
                data=df[[col]].dropna()
                km=KMeans(n_clusters=3,n_init=2,n_jobs=-1,max_iter=10,tol=0.0001)
                df[col]=pd.Series(km.fit_predict(data),index=data.index)
                df[col]=df[col].astype('str')

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

#参数调优
def rscv(x,y,clf,param_grid):
        rs=RandomizedSearchCV(clf,param_distributions=para,n_iter=10,cv=3,scoring='roc_auc',n_jobs=-1)
        rs.fit(x,y)
        print(rs.best_score_,rs.best_params_)
        return rs.best_estimator_

#学习曲线
def plot_learning_curve(x,y,clf):
    sns.set(font_scale=2)
    plt.figure(figsize=(12,8))
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, x, y, cv=3, n_jobs=-1,scoring='accuracy', train_sizes=np.linspace(.2,1.0,5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

#stacking建模
def get_stack_data(clfs, x_train, y_train, x_test):
    stack_train=np.zeros((len(x_train),len(clfs)))
    stack_test=np.zeros((len(x_test),len(clfs)))
    for j, clf in enumerate(clfs):
        oof_test = np.zeros((len(x_test),5))
        skf=StratifiedKFold(n_splits =5,shuffle=False)
        for i,(train_s_index,test_s_index) in enumerate(skf.split(x_train,y_train)):
            x_train_s=x_train.loc[train_s_index]
            y_train_s=y_train.loc[train_s_index]
            x_test_s=x_train.loc[test_s_index]
            clf.fit(x_train_s,y_train_s)
            stack_train[test_s_index,j]=clf.predict_proba(x_test_s)[:,1]
            oof_test[:,i]=clf.predict_proba(x_test)[:,1]
        stack_test[:,j]=oof_test.mean(axis=1)
    return stack_train,stack_test

#1.2数据预处理过程的函数 
#训练集和测试集预处理
def preprocess_train_test():
    #1.导入数据
    train=pd.read_csv('D:/data_project/credit/application_train.csv')
    test=pd.read_csv('D:/data_project/credit/application_test.csv')
    total=pd.concat([train,test],ignore_index=True,sort=False)
    total.columns=total.columns.map(str.lower)
  
    #2.异常值处理
    #code_gender删除XNA的行
    total=total[total.code_gender != 'XNA']
    # days_employed异常值替换为缺失值
    total['days_employed'].replace(365243,np.nan,inplace=True)
    #own_car_age>60的异常替换为缺失值
    total.loc[total.own_car_age>60,'own_car_age']=np.nan
    # amt_income_total离群点数据删除
    '''outliers_index=total[total.amt_income_total>1e7].index
    total.drop(outliers_index,inplace=True)'''

    #3.冗余变量删除
    total=delete_redundancy(total,0.9)

    #4.缺失值填补
    #name_type_suite以'Unaccompanied'填补
    total['name_type_suite']=total['name_type_suite'].fillna('Unaccompanied')
    # cnt_fam_members,使用cnt_children等于0的众数填补缺失列
    cnt_fam_members_isnull=total[total.cnt_fam_members.isnull()]
    fill_class_by_mode(total,'cnt_fam_members','cnt_children',cnt_fam_members_isnull)
    # days_last_phone_change,填补为0
    total['days_last_phone_change']=total['days_last_phone_change'].fillna(0)
    # amt_annuity,ext_source_2,knn回归填补
    cor =total.corr()
    knr=Pipeline([('scl',StandardScaler()),
           ('clf',KNeighborsRegressor(n_neighbors=5, weights='distance',n_jobs=-1))])
    regression_fill(total,cor,'amt_annuity',knr,10)
    regression_fill(total,cor,'ext_source_2',knr,10)
    # own_car_age,回归填补flag_own_car='Y'的样本
    all_columns=abs(cor['own_car_age']).nlargest(10).index.tolist()
    model_columns=[x for x in all_columns if total[x].isnull().sum()==0]
    isnull_index=total[total.own_car_age.isnull()][total.flag_own_car=='Y'].index
    notnull_index=total[total.own_car_age.notnull()].index
    x_train=total.loc[notnull_index,model_columns]
    y_train=total.loc[notnull_index,'own_car_age']
    x_test=total.loc[isnull_index,model_columns]
    knr.fit(x_train,y_train)
    total.loc[isnull_index,'own_car_age']=knr.predict(x_test)

    #5.数值类别变量处理
    #region_rating_client转为object
    total['region_rating_client']=total['region_rating_client'].astype('object')
    # hour_appr_process_start归类
    dict_hour=dict.fromkeys([22,23,0,1,2,3,4,5,6,7],'22-7')
    dict_hour.update(dict.fromkeys(list(range(9,17)),'9-16'))
    dict_hour.update(dict.fromkeys(list(range(17,22)),'17-21'))
    total['hour_appr_process_start']=total['hour_appr_process_start'].replace(dict_hour)

    #6.object类别变量处理
    #emergencystate_mode,name_type_suite
    total['emergencystate_mode']=total.emergencystate_mode.map(lambda x : 0 if x=='No' else 1)
    total['name_type_suite']=total.name_type_suite.map(lambda x : 1 if x=='Unaccompanied' else 0)
    #name_income_type类别合并
    dict1=dict.fromkeys(['Unemployed','Maternity leave'],'high')
    dict1.update(dict.fromkeys(['Student','Businessman'],'low'))
    dict1.update(dict.fromkeys(['State servant','Pensioner'],'state'))
    total['name_income_type']=total['name_income_type'].replace(dict1)
    #name_education_type类别合并
    dict2=dict.fromkeys(['Secondary / secondary special','Incomplete higher'],'second')
    total['name_education_type']=total['name_education_type'].replace(dict2)
    #name_family_status类别合并
    dict3=dict.fromkeys(['Single / not married','Civil marriage'],'not married')
    total['name_family_status']=total['name_family_status'].replace(dict3)
    #name_housing_type类别合并
    dict4=dict.fromkeys(['With parents','Rented apartment'],'no apartment')
    dict4.update(dict.fromkeys(['House / apartment','Co-op apartment'],'house'))
    total['name_housing_type']=total['name_housing_type'].replace(dict4)
    #occupation_type类别合并
    train=total[total.target.notnull()]
    a=train.groupby('occupation_type').target.mean()
    types1=a[a>0.05][a<0.08].index.tolist()
    types2=a[a>=0.08][a<0.15].index.tolist()
    dict5=dict.fromkeys(types1,'advanced_staff')
    dict5.update(dict.fromkeys(types2,'base_staff'))
    total['occupation_type']=total['occupation_type'].replace(dict5)
    #organization_type类别合并
    a=train.groupby('organization_type').target.mean()
    types1=a[a>0.12].index.tolist()
    types2=a[a>0.1][a<=0.12].index.tolist()
    types3=a[a>0.08][a<=0.1].index.tolist()
    types4=a[a>0.06][a<=0.08].index.tolist()
    types5=a[a<=0.06].index.tolist()
    types=[types1,types2,types3,types4,types5]
    dict6={}
    for i in range(5):
        dict6.update(dict.fromkeys(types[i],str(i)))
    total['organization_type']=total['organization_type'].replace(dict6)

    #7.构造3个新特征
    total['income/family']=total['amt_income_total']/total['cnt_fam_members']
    total['income/credit']=total['amt_income_total']/total['amt_credit']
    total['annuity/credit']=total['amt_annuity']/total['amt_credit']

    #8.连续数值变量处理
    # bureau,cnt系列变量,自定义分组
    total['cnt_fam_members']=total.cnt_fam_members.map(lambda x: 0 if x==2 else 1)
    total['cnt_children']=total.cnt_children.map(lambda x: 0 if x==0 else 1)
    total['obs_30_cnt_social_circle']=pd.cut(total.obs_30_cnt_social_circle,[-1,1,6,11,500],labels=list(range(4))).astype('object')
    total['def_30_cnt_social_circle']=pd.cut(total.def_30_cnt_social_circle,[-1,0,1,2,3,50],labels=list(range(5))).astype('object')
    total['def_60_cnt_social_circle']=pd.cut(total.def_60_cnt_social_circle,[-1,0,1,2,3,50],labels=list(range(5))).astype('object')
    total['amt_req_credit_bureau_day']=total.amt_req_credit_bureau_day.map(lambda x : 1 if x>0 else x).astype('object')
    total['amt_req_credit_bureau_mon']=pd.cut(total.amt_req_credit_bureau_mon,[-1,0,1,500],labels=list(range(3))).astype('object')
    total['amt_req_credit_bureau_qrt']=pd.cut(total.amt_req_credit_bureau_qrt,[-1,0,1,500],labels=list(range(3))).astype('object')
    total['amt_req_credit_bureau_year']=pd.cut(total.amt_req_credit_bureau_year,[-1,1,3,500],labels=list(range(3))).astype('object')
    total['amt_req_credit_bureau_hour']=total.amt_req_credit_bureau_hour.map(lambda x : 1 if x>=0 else 0)
    total['amt_req_credit_bureau_week']=total.amt_req_credit_bureau_week.map(lambda x : 1 if x>=0 else 0)
    
    #9.存在缺失值的object特征填补为'Na'
    missing=missing_values(total)
    miss_object=missing[missing.dtype=='object'].index
    total[miss_object]=total[miss_object].fillna('Na')
    return total


#征信局数据预处理
def preprocess_bureau():
    bureau=pd.read_csv('D:/data_project/credit/bureau.csv')
    bb=pd.read_csv('D:/data_project/credit/bureau_balance.csv')
    #1.bureau_balance处理
    bb,bb_cols=one_hot_encoder(bb)#bureau_balance数据独热化
    #聚合
    bb_aggregations={'MONTHS_BALANCE':['min','max','size']}
    for col in bb_cols:
        bb_aggregations[col]='mean'
    bb_agg=bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns=[x[0]+'_'+x[1] for x in list(bb_agg)]  #列名修改
    #合并入bureau数据集
    bureau=pd.merge(bureau,bb_agg,left_on='SK_ID_BUREAU',right_index=True,how='left')
    #2.bureau处理
    bureau.drop(['SK_ID_BUREAU'],axis=1,inplace=True)
    bureau,bureau_cols=one_hot_encoder(bureau)#bureau数据独热化
    #聚合
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
    return bureau_agg

#信用卡消费数据预处理
def preprocess_credit_card():
    credit_card=pd.read_csv('D:/data_project/credit/credit_card_balance.csv')
    credit_card.drop('SK_ID_PREV',axis=1,inplace=True)#删除'SK_ID_PREV'
    credit_card,object_cols=one_hot_encoder(credit_card)#独热化
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
    return cc_agg

#历史贷款数据预处理
def preprocess_previous_application():
    pre_apply=pd.read_csv('D:/data_project/credit/previous_application.csv')
    pre_apply.drop('SK_ID_PREV',axis=1,inplace=True)#删除'SK_ID_PREV'
    #异常值处理
    abnormal_cols=list(pre_apply)[-6:-1]
    for col in abnormal_cols:
        pre_apply[col].replace(365243,np.nan,inplace=True)
    pre_apply,object_cols=one_hot_encoder(pre_apply)#独热化
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
    return pre_agg
    
#pos和现金消费数据预处理
def preprocess_pos_cash():    
    pos=pd.read_csv('D:/data_project/credit/POS_CASH_balance.csv')
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
    #删除冗余变量
    pos_agg=delete_redundancy(pos_agg,0.9)
    return pos_agg

#payments数据预处理
def preprocess_payments():
    pay=pd.read_csv('D:/data_project/credit/installments_payments.csv')
    pay.drop(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'],axis=1,inplace=True)
    #创建分期支付率和提前还款时间特征   
    pay['pay_rate']=pay['AMT_PAYMENT']/pay['AMT_INSTALMENT']
    pay['pay_diff']=pay['DAYS_ENTRY_PAYMENT']-pay['DAYS_INSTALMENT']
    pay['pay_rate']=pay['pay_rate'].map(lambda x:1 if x>1 else x)#处理异常
    #聚合
    pay_aggregations={'NUM_INSTALMENT_VERSION':'nunique'}
    for col in list(pay)[2:]:
        pay_aggregations[col]=['min','max','mean','sum','var']
    pay_agg=pay.groupby('SK_ID_CURR').agg(pay_aggregations)
    pay_agg.columns=[x[0]+'_'+x[1] for x in list(pay_agg)]  #列名修改
    pay_agg['pay_count']=pay.groupby('SK_ID_CURR').size()   #提取记录次数
    #删除冗余变量
    pay_agg=delete_redundancy(pay_agg,0.9)
    return pay_agg
    
#合并所有数据,删除冗余和低信息变量
def merge_all_data():
    df=total
    for new_df in [bureau_agg,pre_agg,pay_agg,pos_agg,cc_agg]:
        df=pd.merge(df,new_df,left_on='sk_id_curr',right_on='SK_ID_CURR',how='left',sort=False)
    df.drop('sk_id_curr',axis=1,inplace=True)
    #填补方差为0变量的缺失值
    value_number=df.nunique()
    cols=value_number[value_number==1].index.tolist()
    df[cols]=df[cols].fillna(1)
    #删除缺失率超过90%的变量
    na_rate=missing_values(df)
    high_na=na_rate[na_rate.NAN_rate>0.9].index.tolist()
    df.drop(high_na,axis=1,inplace=True)
    #删除低互信息特征
    train=df[df.target.notnull()]
    mi=mutual_info(train,'target')
    df=df[mi[mi>0.0001].index]
    #删除冗余特征
    df=delete_redundancy(df,0.9)
    return df



#二、数据预处理
#2.1获取各数据集并预处理
total=preprocess_train_test()#主要的训练集合测试集
bureau_agg=preprocess_bureau()#信用局数据
cc_agg=preprocess_credit_card()#信用卡消费数据
pre_agg=preprocess_previous_application()#贷款历史数据
pos_agg=preprocess_pos_cash()#pos和现金消费数据
pay_agg=preprocess_payments()#payments数据
full=merge_all_data()#合并各数据至主表
del bureau_agg,pre_agg,pay_agg,pos_agg,cc_agg,total

#full.to_csv("d:/data_project/credit/data2.csv", index = False)
cols=['obs_30_cnt_social_circle','def_30_cnt_social_circle','def_60_cnt_social_circle','amt_req_credit_bureau_day',
      'amt_req_credit_bureau_mon','amt_req_credit_bureau_qrt','organization_type','region_rating_client']
dict1={}
for col in cols:dict1[col]=str
full=pd.read_csv('D:/data_project/credit/data2.csv',dtype=dict1)


#2.2下采样，拆分训练集和测试集
full_train=full[full.target.notnull()]
train_under=undersampling(full_train,'target').reset_index(drop=True)


#三、建模并预测
#3.1 ligntGBM模型
#拆分训练集和测试集
x_train,y_train=train_under.iloc[:,1:],train_under.iloc[:,0]
x_test=full[full.target.isnull()].iloc[:,1:]

#object特征转换为category
object_cols=x_train.dtypes[x_train.dtypes=='object'].index
x_train[object_cols]=x_train[object_cols].astype('category')
x_test[object_cols]=x_test[object_cols].astype('category')

#建模，交叉验证
lgb=LGBMClassifier(objective='binary',
                   learning_rate=0.05,
                   n_estimators=200,
                   colsample_bytree=0.8,
                   subsample=0.8,
                   num_leaves=30,
                   n_jobs=-1)
para={'max_depth':[6,8,10],'max_bins':[30,50,100]}
%time lgb=rscv(x_train,y_train,lgb,para)
#%time cross_val_score(lgb,x_train,y_train,cv=3,scoring='roc_auc').mean()
%time plot_learning_curve(x_train,y_train,lgb)#学习曲线

#查看特征重要度
importance=pd.Series(lgb.feature_importances_,index=list(x_train)).sort_values(ascending=False)
importance.value_counts().sort_index(ascending=False)
plt.figure(figsize=(12,15))
importance.head(30)[::-1].plot(kind='barh')
plt.xlabel('feature importance')

#获取stack_train,stack_test
stack_train_lgb,stack_test_lgb=get_stack_data([lgb], x_train, y_train, x_test)


#3.2 logisticregression
#合并训练和测试数据
x_train=train_under.iloc[:,1:]
x_test=full[full.target.isnull()].iloc[:,1:]
x_total=pd.concat([x_train,x_test],sort=False)

#连续特征离散化
x_total=x_total[importance[importance>5].index]#选择特征重要度大于0的特征
int_continue=['days_birth','days_id_publish']
float_cols=x_total.dtypes[x_total.dtypes=='float64'].index.tolist()
continue_cols=int_continue+float_cols
discretize_variate(x_total,continue_cols)#连续特征离散化,等频+聚类分组
x_total,cols=one_hot_encoder(x_total)#独热化

#pca降维
pca=PCA(n_components=0.99)
x_total_pca=pd.DataFrame(pca.fit_transform(x_total))

#拆分训练和测试数据
x_train=x_total_pca[:train_under.shape[0]]
x_test=x_total_pca[train_under.shape[0]:]

#建模，交叉验证
lr=Pipeline([('scl',StandardScaler()),
             ('clf',LogisticRegression(n_jobs=-1))]) 
para={'clf__C':np.logspace(-2,0,3),'clf__penalty':['l1','l2']}
%time lr=rscv(x_train,y_train,lr,para)

#获取stack_train,stack_test
stack_train_lr,stack_test_lr=get_stack_data([lr], x_train, y_train, x_test)


#3.3 stacking建模并最终预测
#获取合并后的stack_train,stack_test
stack_train=np.concatenate((stack_train_lgb,stack_train_lr),axis=1)
stack_test=np.concatenate((stack_test_lgb,stack_test_lr),axis=1)

#训练第二层logistic
stack_lr=Pipeline([('scl',StandardScaler()),
             ('clf',LogisticRegression(n_jobs=-1))]) 
para={'clf__C':np.logspace(-2,2,3),'clf__penalty':['l1','l2']}
stack_lr=rscv(stack_train,y_train,stack_lr,para)

#预测
stack_lr.fit(stack_train,y_train)
pred=stack_lr.predict_proba(stack_test)[:,1]

#输出结果
sample=pd.read_csv('D:/data_project/credit/sample_submission.csv')
sample['TARGET']=pred
sample.to_csv('d:/data_project/credit/pred20190715.csv', index=False,sep=',')



