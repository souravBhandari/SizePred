#Size Calculator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("Size Data.csv")

#Convert Categories to Numbers
brand_encoder=LabelEncoder()
brand_encoder.fit(data['Brand'])
data['Brand']=brand_encoder.transform(data['Brand'])

#Split the Data in Features and Labels
features=data[ ['Height','Weight','Brand'] ]
labels=data['Fit']

#Split the Features and Labels in Training and Test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)


#Classifier
model=DecisionTreeClassifier()
model.fit(train_features,train_labels)

#Accuracy
accuracy=model.score(test_features,test_labels)
print("Accuracy : ",accuracy*100)

#Prediction
height=float(input("Enter Height : "))
weight=float(input("Enter Weight : "))
brand=input("Enter Brand Name : ")

pred_data=pd.Series([height,weight,brand], index=['Height','Weight','Brand'])
pred_data['Brand']=brand_encoder.transform(pd.Series(pred_data['Brand']))[0]

pred_data=np.array(pred_data)
pred_data.shape=(1,-1)

pred_result=model.predict(pred_data)

print("Your Best Fit will be : ", pred_result[0])
