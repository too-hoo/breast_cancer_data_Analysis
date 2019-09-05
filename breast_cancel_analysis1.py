import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

# 导入Z-Score数据变换
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# 导入数据划分
from sklearn.model_selection import train_test_split


# 加载数据集，需要把数据放到目录中
train_data = pd.read_csv("./data.csv")
# 数据探索
# 因为数据集中列比较多，我们需要把DataFrame中的列全部显示出来
pd.set_option("display.max_columns",None)
print(train_data.columns)
print(train_data.head(5))
print(train_data.describe())

# 将特征字段分成3组
features_mean= list(train_data.columns[2:12])
features_se= list(train_data.columns[12:22])
features_worst= list(train_data.columns[22:32])
# 数据清洗
# ID列没有用，删除该列
train_data.drop("id", axis=1,inplace=True)
# 将 B 良性替换成为 0，M 恶性替换成为 1
train_data['diagnosis'] = train_data['diagnosis'].map({'M':1,'B':0})

# 将肿瘤诊断结果可视化,计算出良性和恶性的样本个数并显示出来。
sns.countplot(train_data['diagnosis'],label="Count")
plt.show()

# 使用热力图呈现features_mean 字段之间的相关性
corr = train_data[features_mean].corr()
plt.figure(figsize=(14,14))
# annot = True 显示每个方格的数据
sns.heatmap(corr,annot=True)
plt.show()

# 特征选择
features_remain = ['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean']

# 抽取30%的数据作为测试集，其余的作为训练集
train,test = train_test_split(data, test_size = 0.3) # in this out main data is splited into train and test 
# 抽取特征选择的数据作为训练和测试的数据
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_y = test['diagnosis']

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

# 创建SVM分类器
model = svm.SVC()
# 用训练集做训练
model.fit(train_X, train_y)
# 用测试集做预测
prediction = model.predict(test_X)
print('默认SVC训练模型测试集准确率（6特征变量）：',metrics.accuracy_score(prediction,test_y))

# =============================================（10特征值）=======================
# 特征选择
features_remain1 = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']

# 抽取30%的数据作为测试集，其余的作为训练集
train,test = train_test_split(data, test_size = 0.3) # in this out main data is splited into train and test 
# 抽取特征选择的数据作为训练和测试的数据
train_X = train[features_remain1]
train_y = train['diagnosis']
test_X = test[features_remain1]
test_y = test['diagnosis']

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

# 创建SVM分类器
model = svm.SVC()
# 用训练集做训练
model.fit(train_X, train_y)
# 用测试集做预测
prediction = model.predict(test_X)
print('默认SVC训练模型测试集准确率（10特征变量）：',metrics.accuracy_score(prediction,test_y))