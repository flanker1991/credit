	项目背景
Home Credit公司的业务之一是向客户提供信贷，因此预测客户是否能够偿还贷款是一项重要的需求。
本项目的目标：预测贷款申请人的还款能力。
主要有以下三部分工作：
（1）	探索主要数据集：首先分析数据集的大小，变量种类和数量，是否有重复行，查看了正负样本比例，分析缺失值和异常值情况；进一步区分类别型字符串变量，类别型数值变量和连续型数值变量，分析三类变量的缺失情况，异常情况，冗余情况，查看各类变量取值与客户违约率之间的联系；
（2）	分析并提取其他数据源信息：分析变量类型，处理异常值，对类别变量和数值变量分别采用groupby方法聚合后提取不同特征
（3）	建模：首先对比下采样和不采样的数据预测准确性和计算耗时，选择下采样方法。建立第一个模型lightGBM，基于模型找出重要特征，对重要特征进行离散化处理后建立logistic回归，对两个模型的预测结果进行stacking集成，完成最终建模。
