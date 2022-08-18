import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# train data
df = pd.read_csv('train.csv')
#df.head()

#dataset shape

'''df.shape'''



#cleaning the data

df.isnull().sum()


#checking for null or missing values using statistics

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())


#check for null values again for confirmation so as to not miss any blanks

'''df.isnull().sum()'''

#drop any missing values
df.dropna(inplace=True)


# data analysis
# comparing different parameters using axelplot



'''plt.figure(figsize = (100, 50))
sns.set(font_scale = 5)
plt.subplot(331)
sns.countplot(df['Gender'],hue=df['Loan_Status'])

plt.subplot(332)
sns.countplot(df['Married'],hue=df['Loan_Status'])

plt.subplot(333)
sns.countplot(df['Education'],hue=df['Loan_Status'])

plt.subplot(334)
sns.countplot(df['Self_Employed'],hue=df['Loan_Status'])

plt.subplot(335)
sns.countplot(df['Property_Area'],hue=df['Loan_Status'])'''


#to replace all the variable names to numeric form we use binary classifications for all the paramteres
df['Loan_Status'].replace('Y',1,inplace=True)
df['Loan_Status'].replace('N',0,inplace=True)

df.Gender=df.Gender.map({'Male':1,'Female':0})
df['Gender'].value_counts()

df.Married=df.Married.map({'Yes':1,'No':0})
df['Married'].value_counts()

df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
df['Dependents'].value_counts()

df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
df['Education'].value_counts()

df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
df['Self_Employed'].value_counts()


df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
df['Property_Area'].value_counts()

#counting the loan amount after binary classifictaion

'''df['LoanAmount'].value_counts()'''



                        # *** LOGISTIC REGRESSION***

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#splitting data for training and testing

X = df.iloc[1:542,1:12].values
y = df.iloc[1:542,12].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

model = LogisticRegression()
model.fit(X_train,y_train)

lr_prediction = model.predict(X_test)

# tested data
print("y_test",y_test)

#predicted data
print("y_predicted",lr_prediction)

#accuracy of the model

print('Logistic Regression accuracy = ', metrics.accuracy_score(lr_prediction,y_test))








