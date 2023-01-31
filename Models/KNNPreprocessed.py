import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

data=pd.read_csv("AllMerge.csv")
x = data["outcome"].map({"Cured" :0, "Failure with resistance" :1, "Failure" :2}).astype(int)

le = preprocessing.LabelEncoder()
data=data.apply(le.fit_transform)

col="outcome"
x_data= data.loc[:, data.columns != col]
y_data=data.loc[:, data.columns == col]

smt = SMOTE()
X_smote, y_smote = smt.fit_resample(x_data, y_data)
x=X_smote
y=y_smote
y=np.ravel(y)

MinMaxScaler = preprocessing.MinMaxScaler()
X_data_minmax = MinMaxScaler.fit_transform(x_data)
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state = 1)
classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, ypred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, ypred)
print("Classification Report:" ,)
print (result1)
result2 = accuracy_score(y_test,ypred)
print("Accuracy:",result2)

pickle.dump(classifier, open('KNNpreprocessed.pkl','wb'))



