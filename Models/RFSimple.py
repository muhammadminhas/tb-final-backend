import pandas as pd
import numpy as nm  
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

data=pd.read_csv("AllMerge.csv")
col="outcome"
# outcome3=data["outcome"]
outcome2 = data["outcome"].map({"Cured" :0, "Failure with resistance" :1, "Failure" :2}).astype(int)
data["outcome"]=outcome2
x= data.loc[:, data.columns != col]

y=data.loc[:, data.columns == col]

le = preprocessing.LabelEncoder()

data=data.apply(le.fit_transform)
data2=data.copy()
data2=data2.loc[:,data2.columns!="outcome"]
data2=data2.iloc[2191]
x=x.apply(le.fit_transform)
###
###

###
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.20, random_state=0)  
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
###
classifier= RandomForestClassifier(n_estimators= 38, criterion="entropy")  
classifier.fit(x_train, y_train)
###
y_pred= classifier.predict(x_test)
###
X_train, X_val, y_train, y_val = train_test_split(x, y)

# try a big number for n_estimator
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=100)
gbrt.fit(X_train, y_train)

# calculate error on validation set
errors = [mean_squared_error(y_val, y_pred)
 for y_pred in gbrt.staged_predict(X_val)]

bst_n_estimators = nm.argmin(errors) + 1
gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
###
###
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred,output_dict=True)
df = pd.DataFrame(result1).transpose()
df.to_csv("classificationreport.csv")
print("Classification Report:" ,)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

pickle.dump(classifier, open('RFSimple.pkl','wb'))