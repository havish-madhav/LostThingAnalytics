#importing required libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
import datetime 


#loading the dataset using pandas
lt=pd.read_csv("C:/Users/hp/Desktop/LostThings.csv")
lt.head()
lt.info()


#checking the data types of the coloumns
lt.dtypes
#looking for the missing values
lt.isnull()
#using heatmap to visualize the missing values
sn.heatmap(lt.isnull(), yticklabels=False,cbar=False,cmap='viridis')


#creating a function to fill the missing values of year using the admission number field
def input_Year(input_data):
    ADMN_NO=input_data[0]
    Year=input_data[1]
    if(lt.Year.empty):
        if(ADMN_NO[3]==9):
            return 3
        elif(ADMN_NO[3]==8):
            return 2
        else:
            return 1
    else:
        return Year
#filling the missing values of year based on admn no
lt['Year']=lt[['ADMN_NO','Year']].apply(input_Year,axis=1)


#creating barplot to visualize and compare which year students lost morethings
plt.figure(figsize=(10,6))
plt.title("Number of lostthings for each year students")
lt['Year'].value_counts().plot(kind='bar')
plt.xlabel("Year")



#filling the null values in contact number coloumn with the reception phn number
lt.Contact_Number.fillna(8662429299)
#lt.head()


#changing type of Year and contact no columns from float to integer
lt.Year.astype('int64')
lt.Contact_Number.astype('int64')
lt.Time.astype('timestamp')


#changing the sex coloumn to numerical data
lt['SEX']=lt['SEX'].map({'M':1, 'F':0}).astype('int64')



#creating a barplot to visualize which gender people lost more things
plt.figure(figsize=(10,6))
plt.title("gender based lostthings")
lt['SEX'].value_counts().plot(kind='bar')
plt.xlabel("SEX")



#extracting the day and month from date column and hour from the time column
#to do this first we need to convert the date type to string
lt.Date.astype('str')
lt['Month']=pd.DatetimeIndex(lt['Date']).month
lt['Day']=pd.DatetimeIndex(lt['Date']).day
lt['Hour']=lt['Time'].hour
lt.head()


#creating function to categorize hour in a day 
def change_hour(input_hour):
    input_hour[0]=lt['Hour']
    if(lt['Hour']>0 & lt['Hour'<=6]):
        return 1
    elif(lt['Hour']>6 & lt['Hour'<=12]):
        return 2
    elif(lt['Hour']>12 & lt['Hour'<=18]):
        return 3
    else:
        return 4
lt['Hour']=lt['Hour'].apply(change_hour,axis=1)


# creating a function to categorizing the values of place
def change_place(input_place):
    input_place[0]=lt['Place']
    if(lt.Place.isin(['Admin Block','W%'])):
        return 1
    else:
        return 0
lt['Place']=lt['Place'].apply(input_place,axis=1)

#now we can drop the coloumns which are not useful
lt.drop(['Date','Time','Name ','MailID'],axis=1)



#splitting data into train and test parts to predict the place using decissiontree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train,x_test,y_train, y_test = train_test_split(lt.drop(['Place'],axis=1),lt['Place'],test_size=0.3)


#Predicting the values and checking the accuracy
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

#checking the accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




