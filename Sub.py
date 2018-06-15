


#a=[Sports, TV Sports, Exercise, Dining, Museums, Art, Hiking, Gaming, Reading, Tv, Theater, Concerts, Shopping, Yoga, Clubbing]


# data["Wenyi"] = (data["museums"] + data["art"] + data["reading"]+data["concerts"])/4
#
# data["zhai"] = (data["tvsports"] + data["gaming"] + data["tv"])/3
#
# data["lang"]= (data["clubbing"] + data["shopping"] + data["dining"])/3
#
# data["Yundong"] =( data["sports"] + data["yoga"] + data["exercise"] + data[
#     "hiking"])/4.0
import pickle
sports=10
yoga=1
exercise=1
hiking=1
museums=2
art=7
reading=1
concerts=5
tvsports=10
gaming=6
tv=3
clubbing=1
shopping=5
concerts=3
dining=10
YD=(sports+yoga+exercise+hiking)/4
WY=(museums+art+reading+concerts)/4
Zhai=(tvsports+gaming+tv)/3
Lang=(clubbing+shopping+dining)/3

X_test=[[YD,WY,Zhai,Lang]]
file=open("finalized_model.sav","rb")


cl1=pickle.load(file)


selfCluster=cl1.predict(X_test)[0]

def yourparCluster(a):
    if a==0:
        return 3
    if a==1:
        return 1
    if a==2:
        return 2
    if a==3:
        return 0
output=[selfCluster,yourparCluster(selfCluster)]
print(output)# output


