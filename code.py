# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 20:01:25 2018
@author: Arjun
"""
import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
with open("train.json") as datafile:
    data = json.load(datafile)
with open("test.json") as datafile:
    da = json.load(datafile)
df = pd.DataFrame(data)
tdf=pd.DataFrame(da)
df["ingredients"]= [[x.replace(" ","") for x in row] for row in df["ingredients"]]
tdf["ingredients"]= [[x.replace(" ","") for x in row] for row in tdf["ingredients"]]
cuisine_set= set(j for j in df["cuisine"])
numbers_set = set(i for j in df["ingredients"] for i in j)
tdf["ingredients"]=[[ subelt for subelt in elt if subelt in numbers_set ] for elt in tdf["ingredients"]]
mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('ingredients')),
                          columns=mlb.classes_,
                          index=df.index))
tdf=tdf.join(pd.DataFrame(mlb.transform(tdf.pop('ingredients')),
                          columns=mlb.classes_,
                          index=tdf.index))
df["cuisine"]=df["cuisine"].astype('category')
p=dict( enumerate(df['cuisine'].cat.categories) )
df["cuisine"] = df["cuisine"].cat.codes
X=df.iloc[:,3:] 
y = df["cuisine"]
neigh = KNeighborsClassifier(n_neighbors=len(cuisine_set))
neigh.fit(X, y) 
newy=tdf.iloc[:,2:].values;
ypred=neigh.predict(newy)
ypred=[p[x] for x in ypred]
hell=(tdf.iloc[:,0].values);
submission = pd.DataFrame(
    {'id': hell,
     'cuisine': ypred
    })
submission.to_csv('out.csv',index=False)