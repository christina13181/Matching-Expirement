import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import sklearn
import pickle
import scipy
## conda list
## conda info -e
## source activate xxxx
## python --version
## conda install scikit-

Data_all= pd.read_csv('Speed Dating Data.csv',encoding='ISO-8859-1')

print(sklearn.__version__)
data=Data_all[["match","iid","pid","sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"]]

data=data.dropna()

data["Yundong"] =( data["sports"] + data["yoga"] + data["exercise"] + data[
    "hiking"])/4.0
#average
#average
#average
#
data["Wenyi"] = (data["museums"] + data["art"] + data["reading"]+data["concerts"])/4

data["zhai"] = (data["tvsports"] + data["gaming"] + data["tv"])/3

data["lang"]= (data["clubbing"] + data["shopping"] + data["dining"])/3

#
# data["Gw"] = data["theater"] + data["movies"] + data["concerts"] + data[
#     "music"]

cluster1=KMeans(n_clusters=3)
cluster2=KMeans(n_clusters=4)
cluster3=KMeans(n_clusters=5)

data_test=data[["Yundong","Wenyi","zhai","lang"]]
data_test2=data[["Yundong","Wenyi","zhai","lang"]]
data_test3=data[["Yundong","Wenyi","zhai","lang"]]

kmt1=cluster1.fit(data_test)
kmt2=cluster2.fit(data_test2)
kmt3=cluster3.fit(data_test3)

data_test["label1"]=cluster1.labels_
#data.groupby(["music"]).size()/len(data)

test10=data_test.loc[data_test["label1"] == 0]
test11=data_test.loc[data_test["label1"] == 1]
test12=data_test.loc[data_test["label1"] == 2]

data_test2["label2"]=cluster2.labels_#来来来
data["label2"]=cluster2.labels_

test20=data_test2.loc[data_test2["label2"] == 0]
test21=data_test2.loc[data_test2["label2"] == 1]
test22=data_test2.loc[data_test2["label2"] == 2]
test23=data_test2.loc[data_test2["label2"] ==3]
## shuming
datadictionary=data[["iid","label2"]]

Dictionary=datadictionary.drop_duplicates()#  key and value  for iid and label2

history=data[['iid','pid','match']]

history['pid']=history['pid'].astype('int')

test=history.merge(Dictionary,left_on='iid',right_on='iid',how='left')


test.rename(columns={'iid': 'userid'}, inplace=True)

test=test.merge(Dictionary,left_on='pid',right_on='iid',how='left')
test=test[['userid','pid','label2_x','label2_y','match']]

pp=[]

pp00=test[['match']][(test.label2_x==0)&(test.label2_y==0.0)]
pp.append(pp00.mean())

pp01=test[['match']][(test.label2_x==0)&(test.label2_y==1.0)]
pp.append(pp01.mean())

pp02=test[['match']][(test.label2_x==0)&(test.label2_y==2.0)]
pp.append(pp02.mean())

pp03=test[['match']][(test.label2_x==0)&(test.label2_y==3.0)]
pp.append(pp03.mean())

pp11=test[['match']][(test.label2_x==1)&(test.label2_y==1.0)]
pp.append(pp11.mean())

pp12=test[['match']][(test.label2_x==1)&(test.label2_y==2.0)]
pp.append(pp12.mean())

pp13=test[['match']][(test.label2_x==1)&(test.label2_y==3.0)]
pp.append(pp13.mean())

pp22=test[['match']][(test.label2_x==2)&(test.label2_y==2.0)]
pp.append(pp22.mean())

pp23=test[['match']][(test.label2_x==2)&(test.label2_y==3.0)]
pp.append(pp23.mean())

pp33=test[['match']][(test.label2_x==3)&(test.label2_y==3.0)]
pp.append(pp33.mean())



## Shu Ming
data_test3["label3"]=cluster3.labels_
test30=data_test3.loc[data_test3["label3"] == 0]
test31=data_test3.loc[data_test3["label3"] == 1]
test32=data_test3.loc[data_test3["label3"] == 2]
test33=data_test3.loc[data_test3["label3"] ==3]
test34=data_test3.loc[data_test3["label3"] ==4]
def calmean(data):
    cl1=data.mean()["sports"] + data.mean()["yoga"] + data.mean()["exercise"] + data.mean()[
        "hiking"]
    cl2=data.mean()["museums"]+data.mean()["art"]+data.mean()["reading"]

    cl3=data.mean()["tvsports"]+data.mean()["gaming"]+data.mean()["tv"]

    cl4=data.mean()["clubbing"] + data.mean()["shopping"] + data.mean()["dining"]
    cl5=data.mean()["theater"] + data.mean()["movies"] +data.mean()["concerts"] + data.mean()[
        "music"]
    clMeans=[cl1,cl2,cl3,cl4,cl5]
    return clMeans
a1=test10.mean()
b1=test11.mean()
c1=test12.mean()


a2=test20.mean()
b2=test21.mean()
c2=test22.mean()
d2=test23.mean()


a3=test30.mean()
b3=test31.mean()
c3=test32.mean()
d3=test33.mean()
e3=test34.mean()


inertia=[]
label_pred=[]
meanall=data_test
centroids=[]


for k in range(1,10):

    estimator = KMeans(n_clusters=k)#构造聚类器
    estimator.fit(meanall)#聚类
    label_pred.append(estimator.labels_) #获取聚类标签
    centroids.append(estimator.cluster_centers_) #获取聚类中心
    inertia.append(estimator.inertia_ )# 获取聚类准则的总和

print(inertia)
x=range(1,10)
plt.plot(x,inertia)
plt.show()




#scatter
#print(a1,b1,c1)

#print(a2,b2,c2,d2)

big=[a2,b2,c2,d2]
print(big)

big2=[a3,b3,c3,d3,e3]

print(big2)

print(pp)