from fileinput import filename
import imp
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import datetime
from flask_marshmallow import Marshmallow
from sqlalchemy import create_engine
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
import sys
import json
from werkzeug.security import generate_password_hash, check_password_hash

predictionresult="Cannot Predict on Such Data"
data=pd.read_csv("AllMerge.csv")
le = preprocessing.LabelEncoder()
st_x= StandardScaler()
data2=data.copy()
data=data.apply(le.fit_transform)
data3=data.copy()
x=data3.loc[:,data2.columns!="outcome"]
x=st_x.fit_transform(x)
class mlmodel:
    predictionresult="Cannot Predict on Such Data"
    def inverselabel(header,value):
        forinverse=pd.read_csv("AllMerge.csv")
        flage=0
        arrKey=nm.array([],dtype=int)
        arrValue=nm.array([],dtype=object)
        arrHeader=nm.array([],dtype=object)
        
        for x in range(forinverse.columns.size-1):
           ids = le.fit_transform(forinverse[forinverse.columns[x]])
           for y in range (le.classes_.size):
             try:
                arrKey = nm.append(arrKey, y)
                arrValue=nm.append(arrValue,le.classes_[y])
                arrHeader=nm.append(arrHeader,forinverse.columns[x])
             except Exception as e:
                print("Computing")
        print("Predicting Please Wait....")
        for x in range(arrValue.size):
            if(header==arrHeader[x] and value==arrValue[x]):
                flage=1
                return arrKey[x]
            if(x==arrValue.size-1):
                if(flage==0):
                   return -99999

print(mlmodel.inverselabel("period_start",1))

predictor=0
model = pickle.load(open('pickels/RFPreprocessed.pkl','rb'))
KNNPreprocessed = pickle.load(open('pickels/KNNPreprocessed.pkl','rb'))
KNN = pickle.load(open('pickels/KNNSimple.pkl','rb'))
# arrpred=nm.array([],dtype=int)

# data2= data2.loc[:, data2.columns != "outcome"]
# # for x in range(2400):
# #     if(data["outcome"][x]=="Cured"):
# #         print(x)
# print("inverse:",mlmodel.inverselabel(data2.columns[0],"Belarus"))
# print(data2)
# for x in range(data.columns.size-1):
#     arrpred=nm.append(arrpred,mlmodel.inverselabel(data2.columns[x],data2[data2.columns[x]][0]))
 
# print(arrpred)
# np_array = nm.array(arrpred)
# arrpred=np_array.reshape(1, -1)
# arrpred=st_x.transform(arrpred)
# print(model.predict(arrpred))



    # def predict():
    #         data=pd.read_csv("AllMerge.csv")
    #         col="outcome"
    #         le = preprocessing.LabelEncoder()
    #         data=data.apply(le.fit_transform)
    #         x= data.loc[:, data.columns != col]
    #         y=data.loc[:, data.columns == col]
    #         smt = SMOTE()
    #         X_smote, y_smote = smt.fit_resample(x, y)
    #         x=X_smote
    #         y=y_smote
    #         y=nm.ravel(y)
    #         x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.100, random_state=0)  
    #         st_x= StandardScaler()    
    #         x_train= st_x.fit_transform(x_train)    
    #         x_test= st_x.transform(x_test)
    #         classifier= RandomForestClassifier(n_estimators= 38, criterion="entropy")  
    #         classifier.fit(x_train, y_train)
    #         return classifier
    
# classifier=mlmodel.predict()
# xldata=data.copy()
# xldata=xldata.loc[:,xldata.columns!="outcome"]
# xl=xldata.iloc[1]
# for x in range(28):
#     xl[x]=mlmodel.inverselabel(xldata.columns[x],xl[x])

# #arrpred=[2,	5	,7,	1,	0	,13,	2,	4,	1,	1,	3	,4,	3,	2,	11,	747	,33	,0,	0	,1,	0	,0,	2	,1,	144	,46	,33,	1]
# np_array = nm.array(xl)
# xl=np_array.reshape(1, -1)
# #print(arrpred)
# xl=st_x.fit_transform(xl)
# prediction=classifier.predict(xl)
# print("prediction:",prediction)

engine = create_engine('mysql://root:''@localhost/nih',pool_size=1000000, max_overflow=1000000)
app = Flask(__name__)
app.debug = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/nih'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
from flask_cors import CORS


CORS(app)
ma=Marshmallow(app)
db = SQLAlchemy(app)

# @app.teardown_appcontext
# def shutdown_session(exception=None):
#     db.session.remove()

#nih-dataset
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    usertype=db.Column(db.String(80), nullable=False)
    # other fields go here

    def __repr__(self):
        return '<User %r>' % self.username

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data['username']
    password = data['password']
    usertype=data['usertype']
    hashed_password = generate_password_hash(password)

    # check if username already exists
    user = User.query.filter_by(username=username).first()
    if user is not None:
        return 'Username already exists', 400

    # create new user
    new_user = User(username=username, password=hashed_password,usertype=usertype)
    db.session.add(new_user)
    db.session.commit()
    return 'Success', 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']

    user = User.query.filter_by(username=username).first()
    if user is None:
        return jsonify({'error': 'Invalid Username'}), 401

    if check_password_hash(user.password,password ):
        # Return a successful response
        return jsonify({'message': 'Logged in successfully',"usertype":user.usertype,"username":user.username}), 200
    else:
        # Return an error if the password is incorrect
        return jsonify({'error': 'Invalid password'}), 401



class nih_dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    country=db.Column(db.String(255), nullable=True)
    education=db.Column(db.String(255), nullable=True)	
    employment=db.Column(db.String(255), nullable=True)
    case_definition=db.Column(db.String(255), nullable=True)
    type_of_resistance=db.Column(db.String(255), nullable=True)
    x_ray_count=db.Column(db.Integer, nullable=True)
    organization=db.Column(db.String(255), nullable=True)
    affect_pleura=db.Column(db.String(255), nullable=True)
    overall_percent_of_abnormal_volume=	db.Column(db.String(255), nullable=True)
    le_isoniazid=db.Column(db.String(255), nullable=True)
    le_rifampicin=db.Column(db.String(255), nullable=True)	
    le_p_aminosalicylic_acid=db.Column(db.String(255), nullable=True)
    hain_isoniazid=db.Column(db.String(255), nullable=True)	
    hain_rifampicin=db.Column(db.String(255), nullable=True)	
    period_start=db.Column(db.Integer, nullable=True)
    period_end=db.Column(db.Integer, nullable=True)
    period_span=db.Column(db.Integer, nullable=True)
    regimen_count=db.Column(db.Integer, nullable=True)
    qure_peffusion=db.Column(db.String(255), nullable=True)	
    treatment_status=db.Column(db.String(255), nullable=True)
    regimen_drug=db.Column(db.String(255), nullable=True)
    comorbidity=db.Column(db.String(255), nullable=True)
    ncbi_bioproject=db.Column(db.String(255), nullable=True)	
    gene_name=db.Column(db.String(255), nullable=True)
    x_ray_exists=db.Column(db.String(255), nullable=True)
    ct_exists=	db.Column(db.String(255), nullable=True)
    genomic_data_exists=	db.Column(db.String(255), nullable=True)
    qure_consolidation=	db.Column(db.String(255), nullable=True)
    outcome=db.Column(db.String(255), nullable=True)
    
    def __init__(self,country,education,employment,case_definition,type_of_resistance,x_ray_count,organization,affect_pleura,overall_percent_of_abnormal_volume,le_isoniazid,le_rifampicin,	le_p_aminosalicylic_acid,hain_isoniazid,hain_rifampicin,period_start,period_end,period_span,regimen_count,qure_peffusion,treatment_status,regimen_drug,comorbidity,ncbi_bioproject,	gene_name,x_ray_exists,ct_exists,genomic_data_exists,qure_consolidation,outcome):
        self.country=country  
        self.education=education
        self.employment=employment
        self.case_definition=case_definition
        self.type_of_resistance=type_of_resistance
        self.x_ray_count=x_ray_count
        self.organization=organization
        self.affect_pleura=affect_pleura
        self.overall_percent_of_abnormal_volume=overall_percent_of_abnormal_volume
        self.le_isoniazid=le_isoniazid
        self.le_rifampicin=le_rifampicin
        self.le_p_aminosalicylic_acid=le_p_aminosalicylic_acid
        self.hain_isoniazid=hain_isoniazid
        self.hain_rifampicin=hain_rifampicin
        self.period_start=period_start
        self.period_end=period_end
        self.period_span=period_span
        self.regimen_count=regimen_count
        self.qure_peffusion=qure_peffusion
        self.treatment_status=treatment_status
        self.regimen_drug=regimen_drug
        self.comorbidity=comorbidity
        self.ncbi_bioproject=ncbi_bioproject
        self.gene_name=gene_name
        self.x_ray_exists=x_ray_exists
        self.ct_exists=ct_exists
        self.genomic_data_exists=genomic_data_exists
        self.qure_consolidation=qure_consolidation
        self.outcome=outcome
#Main data set of the site
#multipredictionuserimport
class multiplepredictions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    country=db.Column(db.String(255), nullable=True)
    education=db.Column(db.String(255), nullable=True)	
    employment=db.Column(db.String(255), nullable=True)
    case_definition=db.Column(db.String(255), nullable=True)
    type_of_resistance=db.Column(db.String(255), nullable=True)
    x_ray_count=db.Column(db.Integer, nullable=True)
    organization=db.Column(db.String(255), nullable=True)
    affect_pleura=db.Column(db.String(255), nullable=True)
    overall_percent_of_abnormal_volume=	db.Column(db.String(255), nullable=True)
    le_isoniazid=db.Column(db.String(255), nullable=True)
    le_rifampicin=db.Column(db.String(255), nullable=True)	
    le_p_aminosalicylic_acid=db.Column(db.String(255), nullable=True)
    hain_isoniazid=db.Column(db.String(255), nullable=True)	
    hain_rifampicin=db.Column(db.String(255), nullable=True)	
    period_start=db.Column(db.Integer, nullable=True)
    period_end=db.Column(db.Integer, nullable=True)
    period_span=db.Column(db.Integer, nullable=True)
    regimen_count=db.Column(db.Integer, nullable=True)
    qure_peffusion=db.Column(db.String(255), nullable=True)	
    treatment_status=db.Column(db.String(255), nullable=True)
    regimen_drug=db.Column(db.String(255), nullable=True)
    comorbidity=db.Column(db.String(255), nullable=True)
    ncbi_bioproject=db.Column(db.String(255), nullable=True)	
    gene_name=db.Column(db.String(255), nullable=True)
    x_ray_exists=db.Column(db.String(255), nullable=True)
    ct_exists=	db.Column(db.String(255), nullable=True)
    genomic_data_exists=	db.Column(db.String(255), nullable=True)
    qure_consolidation=	db.Column(db.String(255), nullable=True)
   
    def __init__(self,country,education,employment,case_definition,type_of_resistance,x_ray_count,organization,affect_pleura,overall_percent_of_abnormal_volume,le_isoniazid,le_rifampicin,	le_p_aminosalicylic_acid,hain_isoniazid,hain_rifampicin,period_start,period_end,period_span,regimen_count,qure_peffusion,treatment_status,regimen_drug,comorbidity,ncbi_bioproject,	gene_name,x_ray_exists,ct_exists,genomic_data_exists,qure_consolidation):
        self.country=country  
        self.education=education
        self.employment=employment
        self.case_definition=case_definition
        self.type_of_resistance=type_of_resistance
        self.x_ray_count=x_ray_count
        self.organization=organization
        self.affect_pleura=affect_pleura
        self.overall_percent_of_abnormal_volume=overall_percent_of_abnormal_volume
        self.le_isoniazid=le_isoniazid
        self.le_rifampicin=le_rifampicin
        self.le_p_aminosalicylic_acid=le_p_aminosalicylic_acid
        self.hain_isoniazid=hain_isoniazid
        self.hain_rifampicin=hain_rifampicin
        self.period_start=period_start
        self.period_end=period_end
        self.period_span=period_span
        self.regimen_count=regimen_count
        self.qure_peffusion=qure_peffusion
        self.treatment_status=treatment_status
        self.regimen_drug=regimen_drug
        self.comorbidity=comorbidity
        self.ncbi_bioproject=ncbi_bioproject
        self.gene_name=gene_name
        self.x_ray_exists=x_ray_exists
        self.ct_exists=ct_exists
        self.genomic_data_exists=genomic_data_exists
        self.qure_consolidation=qure_consolidation
                                                                            ############    SINGLE USER DATA Prediction   ############
class individualprediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    country=db.Column(db.String(255), nullable=True)
    education=db.Column(db.String(255), nullable=True)	
    employment=db.Column(db.String(255), nullable=True)
    case_definition=db.Column(db.String(255), nullable=True)
    type_of_resistance=db.Column(db.String(255), nullable=True)
    x_ray_count=db.Column(db.Integer, nullable=True)
    organization=db.Column(db.String(255), nullable=True)
    affect_pleura=db.Column(db.String(255), nullable=True)
    overall_percent_of_abnormal_volume=	db.Column(db.String(255), nullable=True)
    le_isoniazid=db.Column(db.String(255), nullable=True)
    le_rifampicin=db.Column(db.String(255), nullable=True)	
    le_p_aminosalicylic_acid=db.Column(db.String(255), nullable=True)
    hain_isoniazid=db.Column(db.String(255), nullable=True)	
    hain_rifampicin=db.Column(db.String(255), nullable=True)	
    period_start=db.Column(db.Integer, nullable=True)
    period_end=db.Column(db.Integer, nullable=True)
    period_span=db.Column(db.Integer, nullable=True)
    regimen_count=db.Column(db.Integer, nullable=True)
    qure_peffusion=db.Column(db.String(255), nullable=True)	
    treatment_status=db.Column(db.String(255), nullable=True)
    regimen_drug=db.Column(db.String(255), nullable=True)
    comorbidity=db.Column(db.String(255), nullable=True)
    ncbi_bioproject=db.Column(db.String(255), nullable=True)	
    gene_name=db.Column(db.String(255), nullable=True)
    x_ray_exists=db.Column(db.String(255), nullable=True)
    ct_exists=	db.Column(db.String(255), nullable=True)
    genomic_data_exists=	db.Column(db.String(255), nullable=True)
    qure_consolidation=	db.Column(db.String(255), nullable=True)
   
    def __init__(self,country,education,employment,case_definition,type_of_resistance,x_ray_count,organization,affect_pleura,overall_percent_of_abnormal_volume,le_isoniazid,le_rifampicin,	le_p_aminosalicylic_acid,hain_isoniazid,hain_rifampicin,period_start,period_end,period_span,regimen_count,qure_peffusion,treatment_status,regimen_drug,comorbidity,ncbi_bioproject,	gene_name,x_ray_exists,ct_exists,genomic_data_exists,qure_consolidation):
        self.country=country  
        self.education=education
        self.employment=employment
        self.case_definition=case_definition
        self.type_of_resistance=type_of_resistance
        self.x_ray_count=x_ray_count
        self.organization=organization
        self.affect_pleura=affect_pleura
        self.overall_percent_of_abnormal_volume=overall_percent_of_abnormal_volume
        self.le_isoniazid=le_isoniazid
        self.le_rifampicin=le_rifampicin
        self.le_p_aminosalicylic_acid=le_p_aminosalicylic_acid
        self.hain_isoniazid=hain_isoniazid
        self.hain_rifampicin=hain_rifampicin
        self.period_start=period_start
        self.period_end=period_end
        self.period_span=period_span
        self.regimen_count=regimen_count
        self.qure_peffusion=qure_peffusion
        self.treatment_status=treatment_status
        self.regimen_drug=regimen_drug
        self.comorbidity=comorbidity
        self.ncbi_bioproject=ncbi_bioproject
        self.gene_name=gene_name
        self.x_ray_exists=x_ray_exists
        self.ct_exists=ct_exists
        self.genomic_data_exists=genomic_data_exists
        self.qure_consolidation=qure_consolidation





#new imports


class newDataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    importedby=db.Column(db.String(255), nullable=True)
    filename=db.Column(db.String(255), nullable=True)
    country=db.Column(db.String(255), nullable=True)
    education=db.Column(db.String(255), nullable=True)	
    employment=db.Column(db.String(255), nullable=True)
    case_definition=db.Column(db.String(255), nullable=True)
    type_of_resistance=db.Column(db.String(255), nullable=True)
    x_ray_count=db.Column(db.Integer, nullable=True)
    organization=db.Column(db.String(255), nullable=True)
    affect_pleura=db.Column(db.String(255), nullable=True)
    overall_percent_of_abnormal_volume=	db.Column(db.String(255), nullable=True)
    le_isoniazid=db.Column(db.String(255), nullable=True)
    le_rifampicin=db.Column(db.String(255), nullable=True)	
    le_p_aminosalicylic_acid=db.Column(db.String(255), nullable=True)
    hain_isoniazid=db.Column(db.String(255), nullable=True)	
    hain_rifampicin=db.Column(db.String(255), nullable=True)	
    period_start=db.Column(db.Integer, nullable=True)
    period_end=db.Column(db.Integer, nullable=True)
    period_span=db.Column(db.Integer, nullable=True)
    regimen_count=db.Column(db.Integer, nullable=True)
    qure_peffusion=db.Column(db.String(255), nullable=True)	
    treatment_status=db.Column(db.String(255), nullable=True)
    regimen_drug=db.Column(db.String(255), nullable=True)
    comorbidity=db.Column(db.String(255), nullable=True)
    ncbi_bioproject=db.Column(db.String(255), nullable=True)	
    gene_name=db.Column(db.String(255), nullable=True)
    x_ray_exists=db.Column(db.String(255), nullable=True)
    ct_exists=	db.Column(db.String(255), nullable=True)
    genomic_data_exists=	db.Column(db.String(255), nullable=True)
    qure_consolidation=	db.Column(db.String(255), nullable=True)
    outcome=db.Column(db.String(255), nullable=True)
    
    def __init__(self,importedby,filename,country,education,employment,case_definition,type_of_resistance,x_ray_count,organization,affect_pleura,overall_percent_of_abnormal_volume,le_isoniazid,le_rifampicin,	le_p_aminosalicylic_acid,hain_isoniazid,hain_rifampicin,period_start,period_end,period_span,regimen_count,qure_peffusion,treatment_status,regimen_drug,comorbidity,ncbi_bioproject,	gene_name,x_ray_exists,ct_exists,genomic_data_exists,qure_consolidation,outcome):
        self.country=country  
        self.education=education
        self.importedby=importedby
        self.filename=filename
        self.employment=employment
        self.case_definition=case_definition
        self.type_of_resistance=type_of_resistance
        self.x_ray_count=x_ray_count
        self.organization=organization
        self.affect_pleura=affect_pleura
        self.overall_percent_of_abnormal_volume=overall_percent_of_abnormal_volume
        self.le_isoniazid=le_isoniazid
        self.le_rifampicin=le_rifampicin
        self.le_p_aminosalicylic_acid=le_p_aminosalicylic_acid
        self.hain_isoniazid=hain_isoniazid
        self.hain_rifampicin=hain_rifampicin
        self.period_start=period_start
        self.period_end=period_end
        self.period_span=period_span
        self.regimen_count=regimen_count
        self.qure_peffusion=qure_peffusion
        self.treatment_status=treatment_status
        self.regimen_drug=regimen_drug
        self.comorbidity=comorbidity
        self.ncbi_bioproject=ncbi_bioproject
        self.gene_name=gene_name
        self.x_ray_exists=x_ray_exists
        self.ct_exists=ct_exists
        self.genomic_data_exists=genomic_data_exists
        self.qure_consolidation=qure_consolidation
        self.outcome=outcome



# class dataset(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(200), nullable=False)
#     age = db.Column(db.Integer, nullable=False)
    
#     def __init__(self,name,age):
#         self.age=age
#         self.name=name
#this will contain nih app new imports
# class newImports(db.Model):
#      id = db.Column(db.Integer, primary_key=True)
#      name = db.Column(db.String(200), nullable=False)
#      age = db.Column(db.Integer, nullable=False)
#      NameofFile= db.Column(db.String(200), nullable=False)

#      def __init__(self,name,age,NameofFile):
#         self.age=age
#         self.name=name
#         self.NameofFile=NameofFile
#this is notifications table
class notifications(db.Model):
     id = db.Column(db.Integer, primary_key=True)
     username = db.Column(db.String(200), nullable=False)
     filename = db.Column(db.String(200), nullable=False)
     status=db.Column(db.String(200), nullable=False)

     def __init__(self,username,filename,status):
        self.username=username
        self.filename=filename
        self.status=status
        
class ClassifierTable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Classifier = db.Column(db.String(80), nullable=False)
    Precision0=db.Column(db.String(80), nullable=False)
    Precision1=db.Column(db.String(80), nullable=False)
    Precision2=db.Column(db.String(80), nullable=False)
    Recall0=db.Column(db.String(80), nullable=False)
    Recall1=db.Column(db.String(80), nullable=False)
    Recall2=db.Column(db.String(80), nullable=False)
    F1Score0=db.Column(db.String(80), nullable=False)
    F1Score1=db.Column(db.String(80), nullable=False)
    F1Score2=db.Column(db.String(80), nullable=False)
    Accuracy=db.Column(db.String(80), nullable=False)
    Support0=db.Column(db.String(80), nullable=False)
    Support1=db.Column(db.String(80), nullable=False)
    Support2=db.Column(db.String(80), nullable=False)
    def __init__(self,Classifier,Precision0,Precision1,Precision2,Recall0,Recall1,Recall2,F1Score0,F1Score1,F1Score2,Accuracy,Support0,Support1,Support2):
        self.Classifier=Classifier
        self.Precision0=Precision0
        self.Precision1=Precision1
        self.Precision2=Precision2
        self.Recall0=Recall0
        self.Recall1=Recall1
        self.Recall2=Recall2
        self.F1Score0=F1Score0
        self.F1Score1=F1Score1
        self.F1Score2=F1Score2
        self.Accuracy=Accuracy
        self.Support0=Support0
        self.Support1=Support1
        self.Support2=Support2

class classificaitonreportwithpreprocessing(db.Model):
     id = db.Column(db.Integer, primary_key=True)
     precision = db.Column(db.String(255), nullable=False)
     recall = db.Column(db.String(255), nullable=False)
     f1score=db.Column(db.String(255), nullable=False)
     support=db.Column(db.String(255), nullable=False)
     accuracy=db.Column(db.String(255), nullable=False)

     def __init__(self,precision,recall,f1score,support,accuracy):
        self.precision=precision
        self.recall=recall
        self.f1score=f1score
        self.support=support
        self.accuracy=accuracy 

class classificaitonreportwithoutpreprocessing(db.Model):
     id = db.Column(db.Integer, primary_key=True)
     precision = db.Column(db.String(255), nullable=False)
     recall = db.Column(db.String(255), nullable=False)
     f1score=db.Column(db.String(255), nullable=False)
     support=db.Column(db.String(255), nullable=False)
     accuracy=db.Column(db.String(255), nullable=False)

     def __init__(self,precision,recall,f1score,support,accuracy):
        self.precision=precision
        self.recall=recall
        self.f1score=f1score
        self.support=support
        self.accuracy=accuracy

class reportCheck(db.Model):
     id = db.Column(db.Integer, primary_key=True)
     reportno=db.Column(db.Integer, nullable=True)

     def __init__(self,reportno):
        self.reportno=reportno

        
        
# @app.route('/add',methods=['POST'])
# def add_newimport():
#     importdata=request.get_json()
    
#     name = request.json['name']
#     age = request.json['age']

#     data = newImports(name,age)
#     db.session.add(data)
#     db.session.commit()
    

class nih_datasetSchema(ma.Schema):
    class Meta:
        fields=('id','country','education','employment','case_definition','type_of_resistance','x_ray_count','organization','affect_pleura','overall_percent_of_abnormal_volume','le_isoniazid','le_rifampicin','le_p_aminosalicylic_acid','hain_isoniazid','hain_rifampicin','period_start','period_end','period_span','regimen_count','qure_peffusion','treatment_status','regimen_drug','comorbidity','ncbi_bioproject','gene_name','x_ray_exists','ct_exists','genomic_data_exists','qure_consolidation','outcome')
nih_dataset_schema=nih_datasetSchema()
nih_datasets_schema=nih_datasetSchema(many=True)

class multiple_predictionSchema(ma.Schema):
    class Meta:
        fields=('id','country','education','employment','case_definition','type_of_resistance','x_ray_count','organization','affect_pleura','overall_percent_of_abnormal_volume','le_isoniazid','le_rifampicin','le_p_aminosalicylic_acid','hain_isoniazid','hain_rifampicin','period_start','period_end','period_span','regimen_count','qure_peffusion','treatment_status','regimen_drug','comorbidity','ncbi_bioproject','gene_name','x_ray_exists','ct_exists','genomic_data_exists','qure_consolidation')
multiple_prediction_schema=multiple_predictionSchema()
multiple_predictions_schema=multiple_predictionSchema(many=True)

class individual_predictionSchema(ma.Schema):
    class Meta:
        fields=('id','country','education','employment','case_definition','type_of_resistance','x_ray_count','organization','affect_pleura','overall_percent_of_abnormal_volume','le_isoniazid','le_rifampicin','le_p_aminosalicylic_acid','hain_isoniazid','hain_rifampicin','period_start','period_end','period_span','regimen_count','qure_peffusion','treatment_status','regimen_drug','comorbidity','ncbi_bioproject','gene_name','x_ray_exists','ct_exists','genomic_data_exists','qure_consolidation')
individual_prediction_schema=individual_predictionSchema()
individual_predictions_schema=individual_predictionSchema(many=True)

class newDatasetSchema(ma.Schema):
    class Meta:
        fields=('id','importedby','filename','country','education','employment','case_definition','type_of_resistance','x_ray_count','organization','affect_pleura','overall_percent_of_abnormal_volume','le_isoniazid','le_rifampicin','le_p_aminosalicylic_acid','hain_isoniazid','hain_rifampicin','period_start','period_end','period_span','regimen_count','qure_peffusion','treatment_status','regimen_drug','comorbidity','ncbi_bioproject','gene_name','x_ray_exists','ct_exists','genomic_data_exists','qure_consolidation','outcome')
newDatasetschema=newDatasetSchema()
newDatasetsschema=newDatasetSchema(many=True)



class datasetSchema(ma.Schema):
    class Meta:
        fields = ('id','name','age')

dataset_schema = datasetSchema()
datasets_schema = datasetSchema(many=True)

class reportSchema(ma.Schema):
    class Meta:
        fields = ('id','reportno')

report_schema = reportSchema()
reports_schema = reportSchema(many=True)
#
class newImportSchema(ma.Schema):
    class Meta:
        fields = ('id','name','age','NameofFile')

newImport_schema = newImportSchema()
newImports_schema = newImportSchema(many=True) 


class notificationsSchema(ma.Schema):
    class Meta:
        fields = ('id','username','filename','status')

notification_schema = notificationsSchema()
notifications_schema = notificationsSchema(many=True)

class ClassifierTableSchema(ma.Schema):
    class Meta:
        fields = ("id","Classifier","Precision0","Precision1","Precision2","Recall0","Recall1","Recall2","F1Score0","F1Score1","F1Score2","Support0","Support1","Support2","Accuracy")
classifier_table_schema= ClassifierTableSchema()
classifier_tables_schema = ClassifierTableSchema(many=True)

class classificationreportSchema(ma.Schema):
    class Meta:
        fields = ('id','precision','recall','f1score','support','accuracy')

classification_report_schema = classificationreportSchema()
classification_reports_schema = classificationreportSchema(many=True)

class classificationreportWPSchema(ma.Schema):
    class Meta:
        fields = ('id','precision','recall','f1score','support','accuracy')

classification_report_WP_schema = classificationreportWPSchema()
classification_reports_WP_schema = classificationreportWPSchema(many=True)



class multipredictionSchema(ma.Schema):
   class Meta:
        fields=('id','country','education','employment','case_definition','type_of_resistance','x_ray_count','organization','affect_pleura','overall_percent_of_abnormal_volume','le_isoniazid','le_rifampicin','le_p_aminosalicylic_acid','hain_isoniazid','hain_rifampicin','period_start','period_end','period_span','regimen_count','qure_peffusion','treatment_status','regimen_drug','comorbidity','ncbi_bioproject','gene_name','x_ray_exists','ct_exists','genomic_data_exists','qure_consolidation','outcome')

multiprediction_Schema=multipredictionSchema()
multipredictions_Schema=multipredictionSchema(many=True)

@app.route("/addnotifications", methods=["POST"], strict_slashes=False)
def add_newnotification():
    username = request.json['username']
    filename = request.json['filename']
    status=request.json['status']

    record = notifications(
		username=username,
		filename=filename,
        status=status
		)
    db.session.add(record)
    db.session.commit()

    return notification_schema.jsonify(record)

@app.route("/addNewModel", methods=["POST"], strict_slashes=False)
def add_newModel():
    classifier=request.json['Classifier']

    record = ClassifierTable(
        Classifier=classifier,
        Precision0=1,
        Precision1=1,
        Precision2=1,
        Recall0=1,
        Recall1=1,
        Recall2=1,
        F1Score0=1,
        F1Score1=1,
        F1Score2=1,
        Support0=1,
        Support1=1,
        Support2=1,
        Accuracy=1
        )
    db.session.add(record)
    db.session.commit()

    return classifier_table_schema.jsonify(record)

    #newimports by user
@app.route("/addreportcheck", methods=["POST"], strict_slashes=False)
def add_newreportcheck():
    guide = reportCheck.query.get(1)
    reportno = request.json['reportno']
    print(reportno)
    guide.reportno = reportno
    db.session.commit()
    
    return reports_schema(guide)

@app.route("/addModel", methods=["POST"], strict_slashes=False)
def add_Model():
    guide = ClassifierTable.query.get(1)
    classifier = request.json['Classifier']
 
    guide.Classifier = classifier
    db.session.commit()
    Precision0=1
    Precision1=1
    Precision2=1
    Recall0=1
    Recall1=1
    Recall2=1
    F1Score0=1
    F1Score1=1
    F1Score2=1
    Support0=1
    Support1=1
    Support2=1
    Accuracy=1
    if(classifier=='1DT'):
        Precision0=0.91
        Precision1=0.96
        Precision2=0.91
        Recall0=0.90
        Recall1=0.97
        Recall2=0.89
        F1Score0=0.90
        F1Score1=0.97
        F1Score2=0.89
        Support0=411
        Support1=427
        Support2=410
        Accuracy=0.90
    elif(classifier=='0DT'):
        Precision0=0.92
        Precision1=0.30
        Precision2=0.64
        Recall0=0.93
        Recall1=0.67
        Recall2=0.49
        F1Score0=0.93
        F1Score1=0.41
        F1Score2=0.56
        Support0=412
        Support1=9
        Support2=73
        Accuracy=0.86
    elif(classifier=='0KNN'):
        Precision0=0.88
        Precision1=0.45
        Precision2=0.56
        Recall0=0.95
        Recall1=0.56
        Recall2=0.30
        F1Score0=0.91
        F1Score1=0.50
        F1Score2=0.39
        Support0=412
        Support1=9
        Support2=73
        Accuracy=0.84
    elif(classifier=='1KNN'):
        Precision0=0.93
        Precision1=0.91
        Precision2=0.84
        Recall0=0.73
        Recall1=0.98
        Recall2=0.95
        F1Score0=0.82
        F1Score1=0.95
        F1Score2=0.89
        Support0=411
        Support1=427
        Support2=410
        Accuracy=0.88
    elif(classifier=='0RF'):
        Precision0=0.91
        Precision1=0.87
        Precision2=0.63
        Recall0=0.98
        Recall1=0.54
        Recall2=0.43
        F1Score0=0.94
        F1Score1=0.67
        F1Score2=0.51
        Support0=412
        Support1=66
        Support2=16
        Accuracy=0.90
    elif(classifier=='1RF'):
        Precision0=0.97
        Precision1=0.97
        Precision2=0.96
        Recall0=0.94
        Recall1=0.96
        Recall2=0.99
        F1Score0=0.95
        F1Score1=0.96
        F1Score2=0.98
        Support0=408
        Support1=421
        Support2=419
        Accuracy=0.96
    guide.Precision0 = Precision0
    guide.Precision1 = Precision1
    guide.Precision2 = Precision2
    guide.Recall0 = Recall0
    guide.Recall1 = Recall1
    guide.Recall2 = Recall2
    guide.F1Score0 = F1Score0
    guide.F1Score1 = F1Score1
    guide.F1Score2 = F1Score2
    guide.Support0 = Support0
    guide.Support1 = Support1
    guide.Support2 = Support2
    guide.Accuracy = Accuracy
    db.session.commit()
    return classifier_table_schema.jsonify(guide)

    
# @app.route("/addmultiprediction", methods=["POST"], strict_slashes=False)
# def add_multiprediction():
#     country=request.json['country']
#     education=	request.json['education']
#     employment= request.json['employment']
#     case_definition= request.json['case_definition']
#     type_of_resistance= request.json['type_of_resistance']
#     x_ray_count= request.json['x_ray_count']
#     organization= request.json['organization']
#     affect_pleura= request.json['affect_pleura']
#     overall_percent_of_abnormal_volume=	 request.json['overall_percent_of_abnormal_volume']
#     le_isoniazid=	 request.json['le_isoniazid']
#     le_rifampicin=	 request.json['le_rifampicin']
#     le_p_aminosalicylic_acid= request.json['le_p_aminosalicylic_acid']
#     hain_isoniazid=	 request.json['hain_isoniazid']
#     hain_rifampicin=	 request.json['hain_rifampicin']
#     period_start= request.json['period_start']
#     period_end= request.json['period_end']
#     period_span= request.json['period_span']
#     regimen_count= request.json['regimen_count']
#     qure_peffusion=	 request.json['qure_peffusion']
#     treatment_status= request.json['treatment_status']
#     regimen_drug= request.json['regimen_drug']
#     comorbidity= request.json['comorbidity']
#     ncbi_bioproject=	 request.json['ncbi_bioproject']
#     gene_name= request.json['gene_name']
#     x_ray_exists= request.json['x_ray_exists']
#     ct_exists=	 request.json['ct_exists']
#     genomic_data_exists=	 request.json['genomic_data_exists']
#     qure_consolidation=	 request.json['qure_consolidation']
#     outcome=request.json["outcome"]
#     # ######
#     # country1=mlmodel.inverselabel("country",country)
#     # education1=mlmodel.inverselabel("education",education)
#     # employment1=mlmodel.inverselabel("employment",employment)
#     # case_definition1=mlmodel.inverselabel("case_definition",case_definition)
#     # type_of_resistance1=mlmodel.inverselabel("type_of_resistance",type_of_resistance)
#     # x_ray_count1=mlmodel.inverselabel("x_ray_count",int(x_ray_count))
#     # organization1=mlmodel.inverselabel("organization",organization)
#     # affect_pleura1=mlmodel.inverselabel("affect_pleura",affect_pleura)
#     # overall_percent_of_abnormal_volume1=mlmodel.inverselabel("overall_percent_of_abnormal_volume",overall_percent_of_abnormal_volume)
#     # le_isoniazid1=mlmodel.inverselabel("le_isoniazid",le_isoniazid)
#     # le_rifampicin1=mlmodel.inverselabel("le_rifampicin",le_rifampicin)
#     # le_p_aminosalicylic_acid1=mlmodel.inverselabel("le_p_aminosalicylic_acid",le_p_aminosalicylic_acid)
#     # hain_isoniazid1=mlmodel.inverselabel("hain_isoniazid",hain_isoniazid)
#     # hain_rifampicin1=mlmodel.inverselabel("hain_rifampicin",hain_rifampicin)
#     # period_start1=mlmodel.inverselabel("period_start",int(period_start))
#     # period_end1=mlmodel.inverselabel("period_end",int(period_end))
#     # period_span1=mlmodel.inverselabel("period_span",int(period_span))
#     # regimen_count1=mlmodel.inverselabel("regimen_count",int(regimen_count))
#     # qure_peffusion1=	mlmodel.inverselabel("qure_peffusion",qure_peffusion)
#     # treatment_status1=mlmodel.inverselabel("treatment_status",treatment_status)
#     # regimen_drug1=mlmodel.inverselabel("regimen_drug",regimen_drug)
#     # comorbidity1=mlmodel.inverselabel("comorbidity",comorbidity)
#     # ncbi_bioproject1=mlmodel.inverselabel("ncbi_bioproject",ncbi_bioproject)
#     # gene_name1=mlmodel.inverselabel("gene_name",gene_name)
#     # x_ray_exists1=mlmodel.inverselabel("x_ray_exists",x_ray_exists)
#     # ct_exists1=mlmodel.inverselabel("ct_exists",ct_exists)
#     # genomic_data_exists1=mlmodel.inverselabel("genomic_data_exists",genomic_data_exists)
#     # qure_consolidation1=mlmodel.inverselabel("qure_consolidation",qure_consolidation)
#     # # print(country1,education1,employment1,case_definition1)
    
#     # # #pred={'country':str(country1),"education":str(education1 ) ,"employment1":str(employment1 ),"case_definition1":str(case_definition1 ),"type_of_resistance1":str(type_of_resistance1 ),"x_ray_count1":str(x_ray_count1 ),"organization1":str( organization1),"affect_pleura1":str( affect_pleura1),"overall_percent_of_abnormal_volume1":str(overall_percent_of_abnormal_volume1 ),"le_isoniazid1":str( le_isoniazid1),"le_rifampicin1":str( le_rifampicin1),"le_p_aminosalicylic_acid1":str(le_p_aminosalicylic_acid1 ),"hain_isoniazid1":str(hain_isoniazid1 ),"hain_rifampicin1":str( hain_rifampicin1),"period_start1":str(period_start1 ),"period_end1":str( period_end1),"period_span1":str(period_span1 ),"regimen_count1":str(regimen_count1 ),"qure_peffusion1":str( qure_peffusion1),"treatment_status1":str(treatment_status1 ),"regimen_drug1":str(regimen_drug1 ),"comorbidity1":str(comorbidity1 ),"ncbi_bioproject1":str(ncbi_bioproject1 ),"gene_name1":str(gene_name1 ),"x_ray_exists1":str( x_ray_exists1),"ct_exists1":str(ct_exists1 ),"genomic_data_exists1":str(genomic_data_exists1 ),"qure_consolidation1":str(qure_consolidation1)}
#     # # ###
#     # predictiondata=pd.read_csv("Multiprediction.csv")
#     # newdf=pd.DataFrame({"country":[country1],"education":[education1],"employment":[employment1],"case_definition":[case_definition1],"type_of_resistance":[type_of_resistance1],"x_ray_count1":[x_ray_count1],"organization": [organization1],"affect_pleura":[affect_pleura1] ,"overall_percent_of_abnormal_volume":[overall_percent_of_abnormal_volume1],"le_isoniazid":[le_isoniazid1],"le_rifampicin":[le_rifampicin1],"le_p_aminosalicylic_acid":[le_p_aminosalicylic_acid1],"hain_isoniazid":[hain_isoniazid1],"hain_rifampicin":[hain_rifampicin1],"period_start":[period_start1],"period_end":[period_end1],"period_span":[period_span1] ,"regimen_count":[regimen_count1] ,"qure_peffusion":[qure_peffusion1] ,"treatment_status":[treatment_status1] ,"regimen_drug":[regimen_drug1] ,"comorbidity":[comorbidity1] ,"ncbi_bioproject":[ncbi_bioproject1] ,"gene_name":[gene_name1] ,"x_ray_exists":[x_ray_exists1] ,"ct_exists":[ct_exists1] ,"genomic_data_exists":[genomic_data_exists1] ,"qure_consolidation":[qure_consolidation1],"outcome":3})
#     # # print(newdf)
#     # predictiondata = predictiondata.append(newdf, ignore_index = True)                
#     # # # for x in range(len(predictiondata)):
#     # # #     if len(predictiondata!=0):
#     # # #        predictiondata.loc[len(predictiondata)-1]=[country1,education1,employment1,case_definition1,type_of_resistance1,x_ray_count1,organization1,affect_pleura1,overall_percent_of_abnormal_volume1,le_isoniazid1,le_rifampicin1,le_p_aminosalicylic_acid1,hain_isoniazid1,hain_rifampicin1,period_start1,period_end1,period_span1,regimen_count1,qure_peffusion1,treatment_status1,regimen_drug1,comorbidity1,ncbi_bioproject1,gene_name1,x_ray_exists1,ct_exists1,genomic_data_exists1,qure_consolidation1,3]
#     # # #     else:
#     # # #         predictiondata.loc[len(predictiondata)-1]=[country1,education1,employment1,case_definition1,type_of_resistance1,x_ray_count1,organization1,affect_pleura1,overall_percent_of_abnormal_volume1,le_isoniazid1,le_rifampicin1,le_p_aminosalicylic_acid1,hain_isoniazid1,hain_rifampicin1,period_start1,period_end1,period_span1,regimen_count1,qure_peffusion1,treatment_status1,regimen_drug1,comorbidity1,ncbi_bioproject1,gene_name1,x_ray_exists1,ct_exists1,genomic_data_exists1,qure_consolidation1,3]

#     # predictiondata.to_csv("Multiprediction.csv")
#     record = multiprediction(
# 		country=country,			
#         education=education	,
# 		employment=employment,
# 		case_definition=case_definition,
# 		type_of_resistance=type_of_resistance,
# 		x_ray_count=x_ray_count,
# 		organization=organization,
# 		affect_pleura=affect_pleura,
# 		overall_percent_of_abnormal_volume=	overall_percent_of_abnormal_volume,
# 		le_isoniazid=	le_isoniazid,
# 		le_rifampicin=le_rifampicin	,
# 		le_p_aminosalicylic_acid=le_p_aminosalicylic_acid,
# 		hain_isoniazid=	hain_isoniazid,
# 		hain_rifampicin=hain_rifampicin	,
# 		period_start=period_start,
# 		period_end=period_end,
# 		period_span=period_span,
# 		regimen_count=regimen_count,
# 		qure_peffusion=qure_peffusion	,
# 		treatment_status=treatment_status,
# 		regimen_drug=regimen_drug,
# 		comorbidity=comorbidity,
# 		ncbi_bioproject=ncbi_bioproject	,
# 		gene_name=gene_name,
# 		x_ray_exists=x_ray_exists,
# 		ct_exists=ct_exists,
# 		genomic_data_exists=genomic_data_exists,
# 		qure_consolidation=qure_consolidation,
#         outcome=outcome
# 		)

#     db.session.add(record)
#     db.session.commit()

#     engine.dispose()

#     return multipredictionSchema.jsonify(record)
#############################


#############################

@app.route("/putclinical", methods=["POST"])
def clinical_update():
    guide = individualprediction.query.get(1)
    
    
    case_definition= request.json['case_definition']
    type_of_resistance= request.json['type_of_resistance']
    x_ray_count= request.json['x_ray_count']

  
    
    period_start= request.json['period_start']
    period_end= request.json['period_end']
    period_span= request.json['period_span']
    regimen_count= request.json['regimen_count']
    
    treatment_status= request.json['treatment_status']
    regimen_drug= request.json['regimen_drug']
    comorbidity= request.json['comorbidity']
   
    x_ray_exists= request.json['x_ray_exists']
    ct_exists=	 request.json['ct_exists']
    genomic_data_exists=	 request.json['genomic_data_exists']
    
    guide.case_definition = case_definition
    guide.type_of_resistance = type_of_resistance
    guide.x_ray_count = x_ray_count
    guide.period_start = period_start
    guide.period_end = period_end
    guide.period_span = period_span
    guide.regimen_count = regimen_count
    guide.treatment_status = treatment_status
    guide.regimen_drug = regimen_drug
    guide.comorbidity = comorbidity
    guide.x_ray_exists = x_ray_exists
    guide.ct_exists = ct_exists
    guide.genomic_data_exists = genomic_data_exists
    
    db.session.commit()
    return individual_predictions_schema(guide)


@app.route("/putxray", methods=["POST"])
def xray_update():
    guide = individualprediction.query.get(1)
    affect_pleura= request.json['affect_pleura']
    overall_percent_of_abnormal_volume=	 request.json['overall_percent_of_abnormal_volume']
    
   
    qure_peffusion=	 request.json['qure_peffusion']
   
    qure_consolidation=	 request.json['qure_consolidation']
    
    guide.affect_pleura = affect_pleura
    guide.overall_percent_of_abnormal_volume = overall_percent_of_abnormal_volume
    guide.qure_peffusion = qure_peffusion
    guide.qure_consolidation = qure_consolidation

    
    
    db.session.commit()
    return individual_predictions_schema(guide)

@app.route("/putdemographic", methods=["POST"])
def demographic_update():
    guide = individualprediction.query.get(1)
    
    
   
    country=request.json['country']
    education=	request.json['education']
    employment= request.json['employment']
    
    organization= request.json['organization']
    
    
    guide.country = country
    guide.education = education
    guide.employment = employment
    guide.organization = organization

    
    
    
    db.session.commit()
    return individual_predictions_schema(guide)


@app.route("/putbacterial", methods=["POST"])
def bacterial_update():
    guide = individualprediction.query.get(1)
    
    
   
   
    le_isoniazid=	 request.json['le_isoniazid']
    le_rifampicin=	 request.json['le_rifampicin']
    le_p_aminosalicylic_acid= request.json['le_p_aminosalicylic_acid']
    hain_isoniazid=	 request.json['hain_isoniazid']
    hain_rifampicin=	 request.json['hain_rifampicin']
 
    ncbi_bioproject=	 request.json['ncbi_bioproject']
    gene_name= request.json['gene_name']
 
    
    
    guide.le_isoniazid = le_isoniazid
    guide.le_rifampicin = le_rifampicin
    guide.le_p_aminosalicylic_acid = le_p_aminosalicylic_acid
    guide.hain_isoniazid = hain_isoniazid
    guide.hain_rifampicin = hain_rifampicin
    guide.ncbi_bioproject = ncbi_bioproject
    guide.gene_name = gene_name



    
    
    
    db.session.commit()
    return individual_predictions_schema(guide)









@app.route("/addmultidata", methods=["POST"], strict_slashes=False)
def add_multiDataset():
    country=request.json['country']
    education=	request.json['education']
    employment= request.json['employment']
    case_definition= request.json['case_definition']
    type_of_resistance= request.json['type_of_resistance']
    x_ray_count= request.json['x_ray_count']
    organization= request.json['organization']
    affect_pleura= request.json['affect_pleura']
    overall_percent_of_abnormal_volume=	 request.json['overall_percent_of_abnormal_volume']
    le_isoniazid=	 request.json['le_isoniazid']
    le_rifampicin=	 request.json['le_rifampicin']
    le_p_aminosalicylic_acid= request.json['le_p_aminosalicylic_acid']
    hain_isoniazid=	 request.json['hain_isoniazid']
    hain_rifampicin=	 request.json['hain_rifampicin']
    period_start= request.json['period_start']
    period_end= request.json['period_end']
    period_span= request.json['period_span']
    regimen_count= request.json['regimen_count']
    qure_peffusion=	 request.json['qure_peffusion']
    treatment_status= request.json['treatment_status']
    regimen_drug= request.json['regimen_drug']
    comorbidity= request.json['comorbidity']
    ncbi_bioproject=	 request.json['ncbi_bioproject']
    gene_name= request.json['gene_name']
    x_ray_exists= request.json['x_ray_exists']
    ct_exists=	 request.json['ct_exists']
    genomic_data_exists=	 request.json['genomic_data_exists']
    qure_consolidation=	 request.json['qure_consolidation']
    

    record = multiplepredictions(
    
		country=country,			
        education=education	,
		employment=employment,
		case_definition=case_definition,
		type_of_resistance=type_of_resistance,
		x_ray_count=x_ray_count,
		organization=organization,
		affect_pleura=affect_pleura,
		overall_percent_of_abnormal_volume=	overall_percent_of_abnormal_volume,
		le_isoniazid=	le_isoniazid,
		le_rifampicin=le_rifampicin	,
		le_p_aminosalicylic_acid=le_p_aminosalicylic_acid,
		hain_isoniazid=	hain_isoniazid,
		hain_rifampicin=hain_rifampicin	,
		period_start=period_start,
		period_end=period_end,
		period_span=period_span,
		regimen_count=regimen_count,
		qure_peffusion=qure_peffusion	,
		treatment_status=treatment_status,
		regimen_drug=regimen_drug,
		comorbidity=comorbidity,
		ncbi_bioproject=ncbi_bioproject	,
		gene_name=gene_name,
		x_ray_exists=x_ray_exists,
		ct_exists=ct_exists,
		genomic_data_exists=genomic_data_exists,
		qure_consolidation=qure_consolidation,
		
		)
    # multipreddataset=pd.read_csv("Multiprediction.csv")
    # print(multipreddataset.head())
    # print(len(multipreddataset))
    # newdf=[len(multipreddataset),country,education,employment,case_definition,type_of_resistance,x_ray_count,organization,affect_pleura,overall_percent_of_abnormal_volume,le_isoniazid,le_rifampicin,le_p_aminosalicylic_acid,hain_isoniazid,hain_rifampicin,period_start,period_end,period_span,regimen_count,qure_peffusion,treatment_status ,regimen_drug,comorbidity ,ncbi_bioproject ,gene_name ,x_ray_exists ,ct_exists ,genomic_data_exists ,qure_consolidation,outcome]
    # print(multipreddataset.head())
    # print(len(multipreddataset))
    # multipreddataset.loc[len(multipreddataset)]=newdf
    # multipreddataset.to_csv("Multiprediction.csv")
    # print(record)
    db.session.add(record)
    db.session.commit()

    engine.dispose()

    return multiple_prediction_schema.jsonify(record)

# @app.route("/addclinical", methods=["POST"], strict_slashes=False)
# def add_clinical():
   
#     case_definition= request.json['case_definition']
#     type_of_resistance= request.json['type_of_resistance']
#     x_ray_count= request.json['x_ray_count']
#     period_start= request.json['period_start']
#     period_end= request.json['period_end']
#     period_span= request.json['period_span']
#     regimen_count= request.json['regimen_count']
   
#     treatment_status= request.json['treatment_status']
#     regimen_drug= request.json['regimen_drug']
#     comorbidity= request.json['comorbidity']
    
#     x_ray_exists= request.json['x_ray_exists']
#     ct_exists=	 request.json['ct_exists']
#     genomic_data_exists=	 request.json['genomic_data_exists']
    
#     # ######
   
#     case_definition1=mlmodel.inverselabel("case_definition",case_definition)
#     type_of_resistance1=mlmodel.inverselabel("type_of_resistance",type_of_resistance)
#     x_ray_count1=mlmodel.inverselabel("x_ray_count",int(x_ray_count))
    
#     period_start1=mlmodel.inverselabel("period_start",int(period_start))
#     period_end1=mlmodel.inverselabel("period_end",int(period_end))
#     period_span1=mlmodel.inverselabel("period_span",int(period_span))
#     regimen_count1=mlmodel.inverselabel("regimen_count",int(regimen_count))
    
#     treatment_status1=mlmodel.inverselabel("treatment_status",treatment_status)
#     regimen_drug1=mlmodel.inverselabel("regimen_drug",regimen_drug)
#     comorbidity1=mlmodel.inverselabel("comorbidity",comorbidity)
#     x_ray_exists1=mlmodel.inverselabel("x_ray_exists",x_ray_exists)
#     ct_exists1=mlmodel.inverselabel("ct_exists",ct_exists)
#     genomic_data_exists1=mlmodel.inverselabel("genomic_data_exists",genomic_data_exists)

   
#     #print(country1)
   
#     #pred={'country':str(country1),"education":str(education1 ) ,"employment1":str(employment1 ),"case_definition1":str(case_definition1 ),"type_of_resistance1":str(type_of_resistance1 ),"x_ray_count1":str(x_ray_count1 ),"organization1":str( organization1),"affect_pleura1":str( affect_pleura1),"overall_percent_of_abnormal_volume1":str(overall_percent_of_abnormal_volume1 ),"le_isoniazid1":str( le_isoniazid1),"le_rifampicin1":str( le_rifampicin1),"le_p_aminosalicylic_acid1":str(le_p_aminosalicylic_acid1 ),"hain_isoniazid1":str(hain_isoniazid1 ),"hain_rifampicin1":str( hain_rifampicin1),"period_start1":str(period_start1 ),"period_end1":str( period_end1),"period_span1":str(period_span1 ),"regimen_count1":str(regimen_count1 ),"qure_peffusion1":str( qure_peffusion1),"treatment_status1":str(treatment_status1 ),"regimen_drug1":str(regimen_drug1 ),"comorbidity1":str(comorbidity1 ),"ncbi_bioproject1":str(ncbi_bioproject1 ),"gene_name1":str(gene_name1 ),"x_ray_exists1":str( x_ray_exists1),"ct_exists1":str(ct_exists1 ),"genomic_data_exists1":str(genomic_data_exists1 ),"qure_consolidation1":str(qure_consolidation1)}
#     ###
#     predictiondata=pd.read_csv("PredictionResult.csv")
#    # predictiondata.loc[0]=[country1,education1,employment1,case_definition1,type_of_resistance1,x_ray_count1,organization1,affect_pleura1,overall_percent_of_abnormal_volume1,le_isoniazid1,le_rifampicin1,le_p_aminosalicylic_acid1,hain_isoniazid1,hain_rifampicin1,period_start1,period_end1,period_span1,regimen_count1,qure_peffusion1,treatment_status1,regimen_drug1,comorbidity1,ncbi_bioproject1,gene_name1,x_ray_exists1,ct_exists1,genomic_data_exists1,qure_consolidation1,3]
#     predictiondata.loc[0,"case_definition"]=case_definition1
#     predictiondata.loc[0,"type_of_resistance"]=type_of_resistance1
#     predictiondata.loc[0,"x_ray_count"]=x_ray_count1
#     predictiondata.loc[0,"period_start"]=period_start1
#     predictiondata.loc[0,"period_end"]=period_end1
#     predictiondata.loc[0,"period_span"]=period_span1
#     predictiondata.loc[0,"regimen_count"]=regimen_count1
#     predictiondata.loc[0,"treatment_status"]=treatment_status1
#     predictiondata.loc[0,"regimen_drug"]=regimen_drug1
#     predictiondata.loc[0,"comorbidity"]=comorbidity1
#     predictiondata.loc[0,"x_ray_exists"]=x_ray_exists1
#     predictiondata.loc[0,"ct_exists"]=ct_exists1
#     predictiondata.loc[0,"genomic_data_exists"]=genomic_data_exists1

 

#     # print(predictiondata)
#     # arrpred= predictiondata.loc[:, data.columns != "outcome"]
#     # np_array = nm.array(arrpred)
#     # arrpred=np_array.reshape(1, -1)
#     # arrpred=st_x.transform(arrpred)
#     # arrpred=model.predict(arrpred)
#     # for x in range(len(predictiondata)):
#     #     predictiondata["outcome"][len(predictiondata)-1]=arrpred[0]
#     predictiondata.to_csv("PredictionResult.csv")
#     # print(predictiondata)
#     result={"form":"done"}
#     return result
# @app.route("/add_demographic", methods=["POST"], strict_slashes=False)
# def add_demographic():
#     country=request.json['country']
#     education=	request.json['education']
#     employment= request.json['employment']
    
#     organization= request.json['organization']
#     predictiondata=pd.read_csv("PredictionResult.csv")
#     # ######
    
#     country1=mlmodel.inverselabel("country",country)
#     education1=mlmodel.inverselabel("education",education)
#     employment1=mlmodel.inverselabel("employment",employment)
    
#     organization1=mlmodel.inverselabel("organization",organization)
#     print(predictiondata)
#     predictiondata=pd.read_csv("ForPredictions.csv")
    
#     predictiondata.loc[0,"country"]=country1
#     predictiondata.loc[0,"education"]=education1
#     predictiondata.loc[0,"employment"]=employment1
#     predictiondata.loc[0,"organization"]=organization1
#     predictiondata.loc[0,"case_definition"]=predictiondata.loc[0,"case_definition"]
#     predictiondata.loc[0,"type_of_resistance"]=predictiondata.loc[0,"type_of_resistance"]
#     predictiondata.loc[0,"x_ray_count"]= predictiondata.loc[0,"x_ray_count"]
#     predictiondata.loc[0,"period_start"]=predictiondata.loc[0,"period_start"]
#     predictiondata.loc[0,"period_end"]=predictiondata.loc[0,"period_end"]
#     predictiondata.loc[0,"period_span"]=predictiondata.loc[0,"period_span"]
#     predictiondata.loc[0,"regimen_count"]=predictiondata.loc[0,"regimen_count"]
#     predictiondata.loc[0,"treatment_status"]=predictiondata.loc[0,"treatment_status"]
#     predictiondata.loc[0,"regimen_drug"]=predictiondata.loc[0,"regimen_drug"]
#     predictiondata.loc[0,"comorbidity"]= predictiondata.loc[0,"comorbidity"]
#     predictiondata.loc[0,"x_ray_exists"]=predictiondata.loc[0,"x_ray_exists"]
#     predictiondata.loc[0,"ct_exists"]= predictiondata.loc[0,"ct_exists"]
#     predictiondata.loc[0,"genomic_data_exists"]=predictiondata.loc[0,"genomic_data_exists"]

#     predictiondata.loc[0,"affect_pleura"]=predictiondata.loc[0,"affect_pleura"]
#     predictiondata.loc[0,"overall_percent_of_abnormal_volume"]=predictiondata.loc[0,"overall_percent_of_abnormal_volume"]
#     predictiondata.loc[0,"qure_peffusion"]= predictiondata.loc[0,"qure_peffusion"]
#     predictiondata.loc[0,"qure_consolidation"]=predictiondata.loc[0,"qure_consolidation"]

#     print(predictiondata)
#     predictiondata.to_csv("PredictionResult.csv")
#     result={"form":"done"}
#     return result

    

# @app.route("/addxray", methods=["POST"], strict_slashes=False)
# def add_xray():
    
#     affect_pleura= request.json['affect_pleura']
#     overall_percent_of_abnormal_volume=	 request.json['overall_percent_of_abnormal_volume']
   
#     qure_peffusion=	 request.json['qure_peffusion']
    
#     qure_consolidation=	 request.json['qure_consolidation']

#     predictiondata=pd.read_csv("PredictionResult.csv")
    
#     affect_pleura1=mlmodel.inverselabel("affect_pleura",affect_pleura)
#     overall_percent_of_abnormal_volume1=mlmodel.inverselabel("overall_percent_of_abnormal_volume",overall_percent_of_abnormal_volume)
   
#     qure_peffusion1=	mlmodel.inverselabel("qure_peffusion",qure_peffusion)
   
#     qure_consolidation1=mlmodel.inverselabel("qure_consolidation",qure_consolidation)
#     predictiondata=pd.read_csv("ForPredictions.csv")
#     predictiondata.loc[0,"affect_pleura"]=affect_pleura1
#     predictiondata.loc[0,"overall_percent_of_abnormal_volume"]=overall_percent_of_abnormal_volume1
#     predictiondata.loc[0,"qure_peffusion"]=qure_peffusion1
#     predictiondata.loc[0,"qure_consolidation"]=qure_consolidation1
#     predictiondata.to_csv("PredictionResult.csv")
#     result={"form":"done"}
#     return result

# @app.route("/addbacterial", methods=["POST"], strict_slashes=False)
# def add_bacterial():
    
#     le_isoniazid=	 request.json['le_isoniazid']
#     le_rifampicin=	 request.json['le_rifampicin']
#     le_p_aminosalicylic_acid= request.json['le_p_aminosalicylic_acid']
#     hain_isoniazid=	 request.json['hain_isoniazid']
#     hain_rifampicin=	 request.json['hain_rifampicin']
#     ncbi_bioproject=	 request.json['ncbi_bioproject']
#     gene_name= request.json['gene_name']
#     # ######
   
#     le_isoniazid1=mlmodel.inverselabel("le_isoniazid",le_isoniazid)
#     le_rifampicin1=mlmodel.inverselabel("le_rifampicin",le_rifampicin)
#     le_p_aminosalicylic_acid1=mlmodel.inverselabel("le_p_aminosalicylic_acid",le_p_aminosalicylic_acid)
#     hain_isoniazid1=mlmodel.inverselabel("hain_isoniazid",hain_isoniazid)
#     hain_rifampicin1=mlmodel.inverselabel("hain_rifampicin",hain_rifampicin)
    
#     ncbi_bioproject1=mlmodel.inverselabel("ncbi_bioproject",ncbi_bioproject)
#     gene_name1=mlmodel.inverselabel("gene_name",gene_name)
   
#     #print(country1)
   
#     #pred={'country':str(country1),"education":str(education1 ) ,"employment1":str(employment1 ),"case_definition1":str(case_definition1 ),"type_of_resistance1":str(type_of_resistance1 ),"x_ray_count1":str(x_ray_count1 ),"organization1":str( organization1),"affect_pleura1":str( affect_pleura1),"overall_percent_of_abnormal_volume1":str(overall_percent_of_abnormal_volume1 ),"le_isoniazid1":str( le_isoniazid1),"le_rifampicin1":str( le_rifampicin1),"le_p_aminosalicylic_acid1":str(le_p_aminosalicylic_acid1 ),"hain_isoniazid1":str(hain_isoniazid1 ),"hain_rifampicin1":str( hain_rifampicin1),"period_start1":str(period_start1 ),"period_end1":str( period_end1),"period_span1":str(period_span1 ),"regimen_count1":str(regimen_count1 ),"qure_peffusion1":str( qure_peffusion1),"treatment_status1":str(treatment_status1 ),"regimen_drug1":str(regimen_drug1 ),"comorbidity1":str(comorbidity1 ),"ncbi_bioproject1":str(ncbi_bioproject1 ),"gene_name1":str(gene_name1 ),"x_ray_exists1":str( x_ray_exists1),"ct_exists1":str(ct_exists1 ),"genomic_data_exists1":str(genomic_data_exists1 ),"qure_consolidation1":str(qure_consolidation1)}
#     ###
#     predictiondata=pd.read_csv("PredictionResult.csv")
#     print(predictiondata)
#    # predictiondata.loc[0]=[country1,education1,employment1,case_definition1,type_of_resistance1,x_ray_count1,organization1,affect_pleura1,overall_percent_of_abnormal_volume1,le_isoniazid1,le_rifampicin1,le_p_aminosalicylic_acid1,hain_isoniazid1,hain_rifampicin1,period_start1,period_end1,period_span1,regimen_count1,qure_peffusion1,treatment_status1,regimen_drug1,comorbidity1,ncbi_bioproject1,gene_name1,x_ray_exists1,ct_exists1,genomic_data_exists1,qure_consolidation1,3]
#     predictiondata.loc[0,"le_isoniazid"]=le_isoniazid1
#     predictiondata.loc[0,"le_rifampicin"]=le_rifampicin1
#     predictiondata.loc[0,"le_p_aminosalicylic_acid"]=le_p_aminosalicylic_acid1
#     predictiondata.loc[0,"hain_isoniazid"]=hain_isoniazid1
#     predictiondata.loc[0,"hain_rifampicin"]=hain_rifampicin1
#     predictiondata.loc[0,"ncbi_bioproject"]=ncbi_bioproject1
#     predictiondata.loc[0,"gene_name"]=gene_name1

#     # print(predictiondata)
#     # arrpred= predictiondata.loc[:, data.columns != "outcome"]
#     # np_array = nm.array(arrpred)
#     # arrpred=np_array.reshape(1, -1)
#     # arrpred=st_x.transform(arrpred)
#     # arrpred=model.predict(arrpred)
#     # for x in range(len(predictiondata)):
#     #     predictiondata["outcome"][len(predictiondata)-1]=arrpred[0]
#     predictiondata.to_csv("PredictionResult.csv")
#     # print(predictiondata)
#     result={"form":"done"}
#     return result
# @app.route("/addform", methods=["POST"], strict_slashes=False)
# def add_form():
#     country=request.json['country']
#     education=	request.json['education']
#     employment= request.json['employment']
#     case_definition= request.json['case_definition']
#     type_of_resistance= request.json['type_of_resistance']
#     x_ray_count= request.json['x_ray_count']
#     organization= request.json['organization']
#     affect_pleura= request.json['affect_pleura']
#     overall_percent_of_abnormal_volume=	 request.json['overall_percent_of_abnormal_volume']
#     le_isoniazid=	 request.json['le_isoniazid']
#     le_rifampicin=	 request.json['le_rifampicin']
#     le_p_aminosalicylic_acid= request.json['le_p_aminosalicylic_acid']
#     hain_isoniazid=	 request.json['hain_isoniazid']
#     hain_rifampicin=	 request.json['hain_rifampicin']
#     period_start= request.json['period_start']
#     period_end= request.json['period_end']
#     period_span= request.json['period_span']
#     regimen_count= request.json['regimen_count']
#     qure_peffusion=	 request.json['qure_peffusion']
#     treatment_status= request.json['treatment_status']
#     regimen_drug= request.json['regimen_drug']
#     comorbidity= request.json['comorbidity']
#     ncbi_bioproject=	 request.json['ncbi_bioproject']
#     gene_name= request.json['gene_name']
#     x_ray_exists= request.json['x_ray_exists']
#     ct_exists=	 request.json['ct_exists']
#     genomic_data_exists=	 request.json['genomic_data_exists']
#     qure_consolidation=	 request.json['qure_consolidation']
#     # ######
#     country1=mlmodel.inverselabel("country",country)
#     education1=mlmodel.inverselabel("education",education)
#     employment1=mlmodel.inverselabel("employment",employment)
#     case_definition1=mlmodel.inverselabel("case_definition",case_definition)
#     type_of_resistance1=mlmodel.inverselabel("type_of_resistance",type_of_resistance)
#     x_ray_count1=mlmodel.inverselabel("x_ray_count",int(x_ray_count))
#     organization1=mlmodel.inverselabel("organization",organization)
#     affect_pleura1=mlmodel.inverselabel("affect_pleura",affect_pleura)
#     overall_percent_of_abnormal_volume1=mlmodel.inverselabel("overall_percent_of_abnormal_volume",overall_percent_of_abnormal_volume)
#     le_isoniazid1=mlmodel.inverselabel("le_isoniazid",le_isoniazid)
#     le_rifampicin1=mlmodel.inverselabel("le_rifampicin",le_rifampicin)
#     le_p_aminosalicylic_acid1=mlmodel.inverselabel("le_p_aminosalicylic_acid",le_p_aminosalicylic_acid)
#     hain_isoniazid1=mlmodel.inverselabel("hain_isoniazid",hain_isoniazid)
#     hain_rifampicin1=mlmodel.inverselabel("hain_rifampicin",hain_rifampicin)
#     period_start1=mlmodel.inverselabel("period_start",int(period_start))
#     period_end1=mlmodel.inverselabel("period_end",int(period_end))
#     period_span1=mlmodel.inverselabel("period_span",int(period_span))
#     regimen_count1=mlmodel.inverselabel("regimen_count",int(regimen_count))
#     qure_peffusion1=	mlmodel.inverselabel("qure_peffusion",qure_peffusion)
#     treatment_status1=mlmodel.inverselabel("treatment_status",treatment_status)
#     regimen_drug1=mlmodel.inverselabel("regimen_drug",regimen_drug)
#     comorbidity1=mlmodel.inverselabel("comorbidity",comorbidity)
#     ncbi_bioproject1=mlmodel.inverselabel("ncbi_bioproject",ncbi_bioproject)
#     gene_name1=mlmodel.inverselabel("gene_name",gene_name)
#     x_ray_exists1=mlmodel.inverselabel("x_ray_exists",x_ray_exists)
#     ct_exists1=mlmodel.inverselabel("ct_exists",ct_exists)
#     genomic_data_exists1=mlmodel.inverselabel("genomic_data_exists",genomic_data_exists)
#     qure_consolidation1=mlmodel.inverselabel("qure_consolidation",qure_consolidation)
#     print(country1)
   
#     pred={'country':str(country1),"education":str(education1 ) ,"employment1":str(employment1 ),"case_definition1":str(case_definition1 ),"type_of_resistance1":str(type_of_resistance1 ),"x_ray_count1":str(x_ray_count1 ),"organization1":str( organization1),"affect_pleura1":str( affect_pleura1),"overall_percent_of_abnormal_volume1":str(overall_percent_of_abnormal_volume1 ),"le_isoniazid1":str( le_isoniazid1),"le_rifampicin1":str( le_rifampicin1),"le_p_aminosalicylic_acid1":str(le_p_aminosalicylic_acid1 ),"hain_isoniazid1":str(hain_isoniazid1 ),"hain_rifampicin1":str( hain_rifampicin1),"period_start1":str(period_start1 ),"period_end1":str( period_end1),"period_span1":str(period_span1 ),"regimen_count1":str(regimen_count1 ),"qure_peffusion1":str( qure_peffusion1),"treatment_status1":str(treatment_status1 ),"regimen_drug1":str(regimen_drug1 ),"comorbidity1":str(comorbidity1 ),"ncbi_bioproject1":str(ncbi_bioproject1 ),"gene_name1":str(gene_name1 ),"x_ray_exists1":str( x_ray_exists1),"ct_exists1":str(ct_exists1 ),"genomic_data_exists1":str(genomic_data_exists1 ),"qure_consolidation1":str(qure_consolidation1)}
#     ###
#     predictiondata=pd.read_csv("ForPredictions.csv")
#     predictiondata.loc[0]=[country1,education1,employment1,case_definition1,type_of_resistance1,x_ray_count1,organization1,affect_pleura1,overall_percent_of_abnormal_volume1,le_isoniazid1,le_rifampicin1,le_p_aminosalicylic_acid1,hain_isoniazid1,hain_rifampicin1,period_start1,period_end1,period_span1,regimen_count1,qure_peffusion1,treatment_status1,regimen_drug1,comorbidity1,ncbi_bioproject1,gene_name1,x_ray_exists1,ct_exists1,genomic_data_exists1,qure_consolidation1,3]
#     print(predictiondata)
#     arrpred= predictiondata.loc[:, data.columns != "outcome"]
#     np_array = nm.array(arrpred)
#     arrpred=np_array.reshape(1, -1)
#     arrpred=st_x.transform(arrpred)
#     arrpred=model.predict(arrpred)
#     for x in range(len(predictiondata)):
#         predictiondata["outcome"][len(predictiondata)-1]=arrpred[0]
#     predictiondata.to_csv("PredictionResult.csv")
#     print(predictiondata)
#     return pred

###########################
@app.route('/delete',methods=['DELETE'])
def delete():
    db.session.query(multiplepredictions).delete()
    db.session.commit()
    return 'Done', 201
########################
@app.route('/deleteindividual',methods=['DELETE'])
def deleteindividual():
    db.session.query(individualprediction).delete()
    db.session.commit()
    return 'Done', 201
@app.route("/useradddata", methods=["POST"], strict_slashes=False)
def add_newDataset():
    filename=request.json['filename'],
    importedby=request.json['importedby'],
    country=request.json['country']
    education=	request.json['education']
    employment= request.json['employment']
    case_definition= request.json['case_definition']
    type_of_resistance= request.json['type_of_resistance']
    x_ray_count= request.json['x_ray_count']
    organization= request.json['organization']
    affect_pleura= request.json['affect_pleura']
    overall_percent_of_abnormal_volume=	 request.json['overall_percent_of_abnormal_volume']
    le_isoniazid=	 request.json['le_isoniazid']
    le_rifampicin=	 request.json['le_rifampicin']
    le_p_aminosalicylic_acid= request.json['le_p_aminosalicylic_acid']
    hain_isoniazid=	 request.json['hain_isoniazid']
    hain_rifampicin=	 request.json['hain_rifampicin']
    period_start= request.json['period_start']
    period_end= request.json['period_end']
    period_span= request.json['period_span']
    regimen_count= request.json['regimen_count']
    qure_peffusion=	 request.json['qure_peffusion']
    treatment_status= request.json['treatment_status']
    regimen_drug= request.json['regimen_drug']
    comorbidity= request.json['comorbidity']
    ncbi_bioproject=	 request.json['ncbi_bioproject']
    gene_name= request.json['gene_name']
    x_ray_exists= request.json['x_ray_exists']
    ct_exists=	 request.json['ct_exists']
    genomic_data_exists=	 request.json['genomic_data_exists']
    qure_consolidation=	 request.json['qure_consolidation']
    outcome= request.json['outcome']
    

    record = newDataset(
        filename=filename,
        importedby=importedby,
		country=country,			
        education=education	,
		employment=employment,
		case_definition=case_definition,
		type_of_resistance=type_of_resistance,
		x_ray_count=x_ray_count,
		organization=organization,
		affect_pleura=affect_pleura,
		overall_percent_of_abnormal_volume=	overall_percent_of_abnormal_volume,
		le_isoniazid=	le_isoniazid,
		le_rifampicin=le_rifampicin	,
		le_p_aminosalicylic_acid=le_p_aminosalicylic_acid,
		hain_isoniazid=	hain_isoniazid,
		hain_rifampicin=hain_rifampicin	,
		period_start=period_start,
		period_end=period_end,
		period_span=period_span,
		regimen_count=regimen_count,
		qure_peffusion=qure_peffusion	,
		treatment_status=treatment_status,
		regimen_drug=regimen_drug,
		comorbidity=comorbidity,
		ncbi_bioproject=ncbi_bioproject	,
		gene_name=gene_name,
		x_ray_exists=x_ray_exists,
		ct_exists=ct_exists,
		genomic_data_exists=genomic_data_exists,
		qure_consolidation=qure_consolidation,
		outcome=outcome
		)

    db.session.add(record)
    db.session.commit()

    engine.dispose()

    return newDatasetschema.jsonify(record)



#post in viewdata
@app.route("/adminadddata", methods=["POST"], strict_slashes=False)
def add_Dataset():
    country=request.json['country']
    education=	request.json['education']
    employment= request.json['employment']
    case_definition= request.json['case_definition']
    type_of_resistance= request.json['type_of_resistance']
    x_ray_count= request.json['x_ray_count']
    organization= request.json['organization']
    affect_pleura= request.json['affect_pleura']
    overall_percent_of_abnormal_volume=	 request.json['overall_percent_of_abnormal_volume']
    le_isoniazid=	 request.json['le_isoniazid']
    le_rifampicin=	 request.json['le_rifampicin']
    le_p_aminosalicylic_acid= request.json['le_p_aminosalicylic_acid']
    hain_isoniazid=	 request.json['hain_isoniazid']
    hain_rifampicin=	 request.json['hain_rifampicin']
    period_start= request.json['period_start']
    period_end= request.json['period_end']
    period_span= request.json['period_span']
    regimen_count= request.json['regimen_count']
    qure_peffusion=	 request.json['qure_peffusion']
    treatment_status= request.json['treatment_status']
    regimen_drug= request.json['regimen_drug']
    comorbidity= request.json['comorbidity']
    ncbi_bioproject=	 request.json['ncbi_bioproject']
    gene_name= request.json['gene_name']
    x_ray_exists= request.json['x_ray_exists']
    ct_exists=	 request.json['ct_exists']
    genomic_data_exists=	 request.json['genomic_data_exists']
    qure_consolidation=	 request.json['qure_consolidation']
    outcome= request.json['outcome']

    record = nih_dataset(
    
		country=country,			
        education=education	,
		employment=employment,
		case_definition=case_definition,
		type_of_resistance=type_of_resistance,
		x_ray_count=x_ray_count,
		organization=organization,
		affect_pleura=affect_pleura,
		overall_percent_of_abnormal_volume=	overall_percent_of_abnormal_volume,
		le_isoniazid=	le_isoniazid,
		le_rifampicin=le_rifampicin	,
		le_p_aminosalicylic_acid=le_p_aminosalicylic_acid,
		hain_isoniazid=	hain_isoniazid,
		hain_rifampicin=hain_rifampicin	,
		period_start=period_start,
		period_end=period_end,
		period_span=period_span,
		regimen_count=regimen_count,
		qure_peffusion=qure_peffusion	,
		treatment_status=treatment_status,
		regimen_drug=regimen_drug,
		comorbidity=comorbidity,
		ncbi_bioproject=ncbi_bioproject	,
		gene_name=gene_name,
		x_ray_exists=x_ray_exists,
		ct_exists=ct_exists,
		genomic_data_exists=genomic_data_exists,
		qure_consolidation=qure_consolidation,
		outcome=outcome
		)

    db.session.add(record)
    db.session.commit()

    engine.dispose()

    return nih_dataset_schema.jsonify(record)

#post in new imports
# @app.route("/add", methods=["POST"], strict_slashes=False)
# def add_newImport():
#     name = request.json['name']
#     age = request.json['age']
#     NameofFile=request.json['NameofFile']

#     record = newImports(
# 		name=name,
# 		age=age,
#         NameofFile=NameofFile,
# 		)
#     db.session.add(record)
#     db.session.commit()
#     db.session.close()
#     engine.dispose() 
#     db.session.remove()
#     return newImport_schema.jsonify(record)
# @app.route('/getnewimports', methods=['GET', 'POST'])
# def getimports():
#     importedby=request.json["importedby"]
#     filename=request.json['filename']
#     data=newDataset.query.filter_by(newDataset.importedby==importedby, newDataset.filename==filename)
#     results=newDatasetsschema.dump(data)
#     return jsonify(results)

@app.route('/getpred', methods=['GET'])
def getpred():
    first_row = ClassifierTable.query.first()
    classifier=first_row.Classifier
    if(classifier=="1RF"):
        print("RF")
        predictor = predictor=pickle.load(open('pickels/RFPreprocessed.pkl','rb'))
    elif(classifier=="0RF"):
        print("RF Not Preprocessed")
        predictor=pickle.load(open('pickels/RFSimple.pkl','rb'))
    elif(classifier=="1KNN"):
        print("KNN Preprocessed ")
        predictor=KNNPreprocessed
    elif(classifier=="0KNN"):
        print("KNN Not Preprocessed")
        predictor=KNN
    elif(classifier=="0DT"):
        predictor=pickle.load(open('pickels/DecisionTreeSimple.pkl','rb'))
        print("Decision Tree Not Preprocessed")
    elif(classifier=="1DT"):
        predictor=pickle.load(open('pickels/DecisionTreePreprocessed.pkl','rb'))
        print("Decision Tree Preprocessed")

    predicteddata=pd.read_csv("PredictionResult.csv")
    arrpred=predicteddata.loc[:,predicteddata.columns!="outcome"]
    print(arrpred.columns)
    np_array = nm.array(arrpred)
    arrpred=np_array.reshape(1, -1)
    arrpred=st_x.transform(np_array)
    
    prediction=predictor.predict(arrpred)
    result={"prediction":str(prediction[0])}
    return jsonify(result)
    
# @app.route("/getmultipred", methods=['GET'])
# def getmultipred():
#     pred=pd.read_csv("Multiprediction.csv")
#     arrpred=pred.copy()
#     arrpred=pred.loc[:,data2.columns!="outcome"]
#     np_array = nm.array(arrpred)
#     arrpred=np_array.reshape(1, -1)
#     arrpred=st_x.transform(arrpred)
#     arrpred=model.predict(arrpred)
#     predictionsjson={}
#     for x in range(len(arrpred)):
#         preds={"prediction:"+str(x):str(arrpred[x])}
#         predictionsjson.update(preds)

#     return predictionsjson
@app.route('/getmultipredictionresult',methods =['GET'])
def getmultiresult():
    all_entries=multiplepredictions.query.all()
    results = multipredictions_Schema.dump(all_entries)
    print(len(results))
    multidata=pd.read_csv("ForPredictions.csv")
    for x in range(len(results)):
        for y in range(28):
          multidata.loc[x,multidata.columns[y]]=results[x][multidata.columns[y]]
    multidata.to_csv("Multiprediction.csv",index=False)
    for x in range(len(results)):
        for y in range(28):
            multidata.loc[x,multidata.columns[y]]=mlmodel.inverselabel(multidata.columns[y],multidata.loc[x,multidata.columns[y]])
    multidata.to_csv("Multiprediction.csv",index=False)
   
    return jsonify(results)
@app.route('/getindividualpredictionresult',methods =['GET'])
def getindividualresult():
    first_row = ClassifierTable.query.first()
    classifier=first_row.Classifier
    if(classifier=="1RF"):
        print("RF")
        predictor = predictor=pickle.load(open('pickels/RFPreprocessed.pkl','rb'))
    elif(classifier=="0RF"):
        print("RF Not Preprocessed")
        predictor=pickle.load(open('pickels/RFSimple.pkl','rb'))
    elif(classifier=="1KNN"):
        print("KNN Preprocessed ")
        predictor=KNNPreprocessed
    elif(classifier=="0KNN"):
        print("KNN Not Preprocessed")
        predictor=KNN
    elif(classifier=="0DT"):
        predictor=pickle.load(open('pickels/DecisionTreeSimple.pkl','rb'))
        print("Decision Tree Not Preprocessed")
    elif(classifier=="1DT"):
        predictor=pickle.load(open('pickels/DecisionTreePreprocessed.pkl','rb'))
        print("Decision Tree Preprocessed")

    all_entries=individualprediction.query.all()
    results = multipredictions_Schema.dump(all_entries)
    multidata=pd.read_csv("ForPredictions.csv")
    for x in range(1):
        for y in range(28):
          multidata.loc[x,multidata.columns[y]]=results[x][multidata.columns[y]]
    for x in range(1):
        for y in range(28):
            multidata.loc[x,multidata.columns[y]]=mlmodel.inverselabel(multidata.columns[y],multidata.loc[x,multidata.columns[y]])
    arrpred=multidata.copy()
    arrpred=arrpred.loc[:,arrpred.columns!="outcome"]
    np_array = nm.array(arrpred)
    arrpred=np_array.reshape(1, -1)
    arrpred=st_x.transform(np_array)
    prediction=predictor.predict(arrpred)
    multidata.to_csv("SinglePrediction.csv",index=False)
    result={"prediction":str(prediction[0])}
    return jsonify(result)


@app.route('/getrusermultidata',methods =['GET'])
def getinputmulti():
    all_entries=multiplepredictions.query.all()
    results = multipredictions_Schema.dump(all_entries)
    return jsonify(results)
@app.route('/getresult',methods =['GET'])
def getresult():
    first_row = ClassifierTable.query.first()
    classifier=first_row.Classifier
    if(classifier=="1RF"):
        print("RF")
        predictor = predictor=pickle.load(open('pickels/RFPreprocessed.pkl','rb'))
    elif(classifier=="0RF"):
        print("RF Not Preprocessed")
        predictor=pickle.load(open('pickels/RFSimple.pkl','rb'))
    elif(classifier=="1KNN"):
        print("KNN Preprocessed ")
        predictor=KNNPreprocessed
    elif(classifier=="0KNN"):
        print("KNN Not Preprocessed")
        predictor=KNN
    elif(classifier=="0DT"):
        predictor=pickle.load(open('pickels/DecisionTreeSimple.pkl','rb'))
        print("Decision Tree Not Preprocessed")
    elif(classifier=="1DT"):
        predictor=pickle.load(open('pickels/DecisionTreePreprocessed.pkl','rb'))
        print("Decision Tree Preprocessed")

    multidata=pd.read_csv("Multiprediction.csv")
    arrpred=multidata.copy()
    arrpred=arrpred.loc[:,arrpred.columns!="outcome"]
    print(arrpred.columns)
    np_array = nm.array(arrpred)
    arrpred=np_array.reshape(1, -1)
    arrpred=st_x.transform(np_array)
    prediction=predictor.predict(arrpred)
    print(prediction)
    for x in range(len(prediction)):
        prediction[x]=str(prediction[x])
    print(prediction)
    list=[]
    for x in range(len(prediction)):
        list.append(str(prediction[x]))
    # db.session.query(multiplepredictions).delete()
    # db.session.commit()
    json_string = json.dumps(list)
   
    # predresult={}
    # for x in range(len(prediction)):
    #     pred={"prediction:"+str(x):str(prediction[x])}
    #     predresult=json.dump(predresult,pred)
    return json_string
    

@app.route('/get',methods =['GET'])
def get_articles():
    all_entries = nih_dataset.query.all()
    results = nih_datasets_schema.dump(all_entries)
    return jsonify(results)

@app.route('/getnewimports',methods =['GET'])
def get_newdataimports():
    all_entries = newDataset.query.all()
    results = newDatasetsschema.dump(all_entries)
    return jsonify(results)



@app.route('/getnotifications',methods =['GET'])
def get_notifications():
    all_entries = notifications.query.all()
    results = notifications_schema.dump(all_entries)
    
    return jsonify(results)

@app.route('/getreport',methods =['GET'])
def get_report1():
    all_entries = classificaitonreportwithpreprocessing.query.all()
    results = classification_reports_schema.dump(all_entries)
    
    return jsonify(results)

@app.route("/getClassificationReport",methods=['GET'])
def get_classification_report():
    all_entries = ClassifierTable.query.all()
    results = classifier_tables_schema.dump(all_entries)
    return jsonify(results)
@app.route('/getreportwp',methods =['GET'])
def get_report0():
    all_entries = classificaitonreportwithoutpreprocessing.query.all()
    results = classification_reports_WP_schema.dump(all_entries)
    
    return jsonify(results)

@app.route('/getreportcheck',methods =['GET'])
def get_reportcheck():
    all_entries = reportCheck.query.all()
    results = reports_schema.dump(all_entries)
    return jsonify(results)

 
@app.route('/update/<id>',methods = ['PUT'])
def update_article(id):
    notify = notifications.query.get(id)

    status = request.json['status']
    

    notify.status = status
    

    db.session.commit()
    return notification_schema.jsonify(notify)


# @app.route('/getnewdata',methods =['GET'])
# def new

if __name__ == '__main__':
    app.run(debug=True)