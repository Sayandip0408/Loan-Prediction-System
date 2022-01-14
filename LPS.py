from flask import Flask ,render_template,request,url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import os

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("home2.html")

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    # Importing the dataset
    df = pd.read_csv('data.csv')

    #Checking for missing values
    df.isnull().sum()
    
    df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
    
    df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].median())
    
    df.isnull().sum()

    df.dropna(inplace=True)

    df.shape

    print(pd.crosstab(df['Property_Area'],df['Loan_Status']))

    sns.countplot(df['Property_Area'],hue=df['Loan_Status'])

    #Converting dependent categorical variable to continous variable.
    
    df['Loan_Status'].replace('N',0,inplace=True)
    df['Loan_Status'].replace('Y',1,inplace=True)
    

    #Changing categorical variables to continous variables.
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    le=LabelEncoder()
    ohe=OneHotEncoder()

    df['Property_Area']=le.fit_transform(df['Property_Area'])
    df['Gender']=le.fit_transform(df['Gender'])
    df['Married']=le.fit_transform(df['Married'])

    
    X=df.drop('Loan_Status',1)
    Y=df['Loan_Status']

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=6)

    log=LogisticRegression()
    log.fit(x_train,y_train)
    log.score(x_test,y_test)
     
    if request.method == 'POST':
        gender= request.form['gender']
        married= request.form['married']
        loan= request.form['loan']
        c_history= request.form['c_history']
        p_area= request.form['p_area']
    
        #creating a json object to hold the data from the form
        input_data=[{'gender':gender,'married':married,'loan':loan,'c_history':c_history,'p_area':p_area}]
        data= pd.DataFrame(input_data)
        categorical_columns=['gender','married','p_area']
        data['loan']=data['loan'].astype(int)
        data['c_history']=data['c_history'].astype(int)
        data[categorical_columns]=data[categorical_columns].apply(le.fit_transform)
        data[categorical_columns]=data[categorical_columns].astype('object')
        my_result = log.predict(data)
    return render_template('result.html', prediction=my_result)

if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)