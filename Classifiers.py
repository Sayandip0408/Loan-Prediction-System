

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
df = pd.read_csv('train.csv')

#Datatypes of the features
df.dtypes

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

plt.title('Correlation Matrix')
sns.heatmap(df.corr(),annot=True)

df2=df.drop(labels=['ApplicantIncome'],axis=1)
df2=df2.drop(labels=['CoapplicantIncome'],axis=1)
df2=df2.drop(labels=['LoanAmount'],axis=1)
df2=df2.drop(labels=['Loan_Amount_Term'],axis=1)
df2=df2.drop(labels=['Loan_ID'],axis=1)

df2.head()

#Changing categorical variables to continous variables.

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder()

df2['Property_Area']=le.fit_transform(df2['Property_Area'])
df2['Dependents']=le.fit_transform(df2['Dependents'])
df2['Gender']=le.fit_transform(df2['Gender'])
df2['Married']=le.fit_transform(df2['Married'])
df2['Education']=le.fit_transform(df2['Education'])
df2['Self_Employed']=le.fit_transform(df2['Self_Employed'])

#df2=pd.get_dummies(df2)

df2.dtypes

'''df2=df2.drop(labels=['Gender_Female'],axis=1)
df2=df2.drop(labels=['Married_No'],axis=1)
df2=df2.drop(labels=['Education_Not Graduate'],axis=1)
df2=df2.drop(labels=['Self_Employed_No'],axis=1)'''

plt.title('Correlation Matrix')
sns.heatmap(df2.corr(),annot=True)

df2=df2.drop('Self_Employed',1)
df2=df2.drop('Dependents',1)
df2=df2.drop('Education',1)
X=df2.drop('Loan_Status',1)
Y=df2['Loan_Status']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

print('Shape of X_train is: ',x_train.shape)
print('Shape of X_test is: ',x_test.shape)
print('Shape of Y_train is: ',y_train.shape)
print('Shape of y_test is: ',y_test.shape)

#---LOGISTIC REGRESSION---

from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)

log.score(x_train,y_train)

#Predicting trest dataset
pred=log.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)

data={'y_test':y_test,'pred':pred}
pd.DataFrame(data=data)

#-----DECISION TREE------

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

clf.score(x_train,y_train)

pred1=clf.predict(x_test)
accuracy_score(y_test,pred1)


#------K-NN------
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of KNN is: ', metrics.accuracy_score(y_pred, y_test))


#------SVM------
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of SVM is: ', metrics.accuracy_score(y_pred, y_test))


#------Random Forest Classification------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of Random Forest Classification is: ', metrics.accuracy_score(y_pred, y_test))


#------K-SVM------
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Measuring Accuracy
from sklearn import metrics
print('The accuracy of K-SVM is: ', metrics.accuracy_score(y_test, y_pred))

