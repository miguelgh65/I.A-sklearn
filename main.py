import pandas as pd

import matplotlib.pyplot as plt





data=pd.read_csv("car.csv",header = None)

data.columns=["price", "maintenance", "n_doors", "capacity", "size_lug" , "safety", "class"]


# decision=data["Decision"].value_counts()
# data["Decision"].value_counts().sort_index(ascending =False)
# print(decision)
#
# decision.plot(kind = 'bar')
# plt.show()
data.sample(3)

data.price.replace(('vhigh','high','med','low'),(4,3,2,1),inplace=True)
data.maintenance.replace(('vhigh','high','med','low'),(4,3,2,1),inplace=True)
data.n_doors.replace(('2','3','4','5more'),(1,2,3,4),inplace=True)
data.capacity.replace(('2','4','more'),(1,2,3),inplace=True)
data.size_lug.replace(("small","med","big"),(1,2,3),inplace= True)
data.safety.replace(("low","med","high"),(1,2,3),inplace = True)

data['class'].replace(('unacc','acc','good','vgood'),(1,2,3,4),inplace= True)

dataset=data.values


X=dataset[:,0:6]
Y= pd.np.asarray(dataset[:, 6], dtype="S6")



print(data.head(5))

from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=0)

tr=tree.DecisionTreeClassifier(max_depth=10)

tr.fit(X_Train,Y_Train)

y_pred=tr.predict(X_Test)
print(y_pred)

score=tr.score(X_Test,Y_Test)

print("Precision: %0.4f" % (score))