import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import os
SDD = pd.read_csv('Speed Dating Data.csv',encoding='ISO-8859-1')
SD1 = SDD[["fun_o","shar_o","sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"]]
#"sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"

SD2 = SDD[["gender","fun_o","shar_o","sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"]]
#print(SD1)
SD3 = SDD[["gender","race_o","fun_o","shar_o","sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"]]
#print(SD1)
SD4 = SDD[["gender","fun_o","shar_o","sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"]]
# print(SD1)

SD=SDD[["fun_o","shar_o"]]

SDD_data=SDD[["sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"]]
#print(SD1)

#SDDtest_data=SDD[["iid","pid","match","sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing","reading","tv","theater","movies","concerts","music","shopping","yoga"]]
#print(SD1)

SD_data = SD.dropna()# delete al the null value

SDD_data=SDD_data.dropna()

#SDDtest_data=SDDtest_data.dropna()

#SDDtest_datanolabel=SDDtest_data[]


SD_test= np.array(SD_data)

SDD_test=np.array(SDD_data)
#set the type as 3
KMT=KMeans(n_clusters=3)#
#bring the data into the model
kmt=KMT.fit(SD_test)

SD_data["label"]=kmt.labels_
SD_data0=SD_data.loc[SD_data["label"] == 0]
SD_data1=SD_data.loc[SD_data["label"] == 1]
SD_data2=SD_data.loc[SD_data["label"] == 2]


#draw the cluster result in a scatter plot
plt.title('test for Kmeans')
plt.rc('font', family='STXihei', size=10)
plt.scatter(SD_data0['fun_o'],SD_data0['shar_o'],50,color='#99CC01',marker='+',linewidth=2,alpha=0.8)
plt.scatter(SD_data1['fun_o'],SD_data1['shar_o'],50,color='#FE0000',marker='+',linewidth=2,alpha=0.8)
plt.scatter(SD_data2['fun_o'],SD_data2['shar_o'],50,color='#0000FE',marker='+',linewidth=2,alpha=0.8)
plt.xlabel('fun_o')
plt.ylabel('shar_o')
#plt.xlim(0,25000)
plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='both',alpha=0.4)


from sklearn.externals import joblib
KMD=KMeans(n_clusters=3)#
#bring the data into the model
kmd=KMD.fit(SDD_test)

SDD_data["label"]=kmd.labels_
SDD_data0=SDD_data.loc[SDD_data["label"] == 0]
SDD_data1=SDD_data.loc[SDD_data["label"] == 1]
SDD_data2=SDD_data.loc[SDD_data["label"] == 2]


SDD_data0.describe()
SDD_data0.mean()

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
#SDD_data0.mean().["sports"]+SDD_data0.mean().["yoga"]+SDD_data0.mean().["exercise"]+SDD_data0.mean().["hiking"]
print(calmean(SDD_data0))

os.chdir("~/PycharmProjects/SpeedDating/model_save")

joblib.dump(kmd, "train_model.m")

