# machine-learning-prediction-model

this project aims to create a machine learning prediction model to predict which passengers survived from Titanic

I'm going to use Google Colab 

first step is to insert ```Titanic_ML(task)```  file into google colab then upload ```train.csv``` file 

## Importing the Dependencies 

first, i'm going to import the dependencies and run it 

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Reading the data

next it will be reading the data and run them  

```
data = pd.read_csv('/content/train.csv')
```

to make sure the data is imported I'm going to display the 5 rows 

```
data.head()
```

![image](https://github.com/user-attachments/assets/27c2dfec-f69e-46eb-87fb-25f4d8e8caa5)


## Data preprocessing 

to know more about the data we have to run 

```
data.info()
```

![image](https://github.com/user-attachments/assets/4635f361-db5f-4288-bc49-d8aa2b0e132e)


## Dealing with missing data

to deal with messing data we have 177 missing data in the age section, 687 in the cabin section, and 2 in the embarked section  

for the age section, I'm going to complete the data by finding the mean 

```
data['Age'].fillna(data['Age'].mean(), inplace=True)
```

for the cabin section I'm going to drop it 

```
data = data.drop(['Cabin'], axis=1)
```

for the embarked section, i'm going to complete it by the mode

```
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
```

## Drob useless columns 

I'm going to drop the passenger ID, Name, and the Tickets column since they are useless 

```
data = data.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1)
```

## Encode categorical  columns 

next, i'm going to change the gender with 0 for male and 1 for female and the embarked with 0 for S, 1 for C, and 2 for Q 

```
data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
```

![image](https://github.com/user-attachments/assets/6c0cb822-cba9-420f-82d8-f9e3da840d7c)



## Dealing with duplicates 

to check the duplicates and delete them i used 

```
#check if there are duplicates in the dataset:
duplicates = data.duplicated().sum()

#drop the duplicates:
data = data.drop_duplicates()
```

## Data analysis













