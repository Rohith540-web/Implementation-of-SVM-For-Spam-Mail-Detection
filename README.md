# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Start

Import necessary libraries

Import pandas for data manipulation.

Import necessary functions from sklearn for data processing, modeling, and evaluation.

Load the dataset

Read the spam.csv file using pandas.read_csv() with appropriate encoding (Windows-1252).

Display the dataset to verify it is loaded correctly.

Understand the dataset

Check the shape (number of rows and columns) of the dataset.

Extract the input texts (x) from column v2 (email text).

Extract the labels (y) from column v1 (ham/spam).

Split the dataset

Use train_test_split() to split the data into training and testing sets.

Set test size to 20% (test_size=0.2).

Use a random state for reproducibility (random_state=0).

Convert text data to numerical vectors

Initialize CountVectorizer() to convert text to feature vectors.

Fit the vectorizer on training data and transform both training and testing sets.

Train the SVM classifier

Initialize the Support Vector Classifier (SVC()).

Train the classifier using fit() with the vectorized training data and corresponding labels.

Make predictions

Predict labels for the test data using the trained SVM model.

Evaluate the model

Calculate the accuracy using accuracy_score().

Generate a confusion matrix using confusion_matrix().

Display the classification report using classification_report().

Print evaluation results

Print accuracy, confusion matrix, and classification report to analyze model performance.

End



## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: ROHITH V
RegisterNumber:212224220083  


import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
Head():

![Screenshot 2025-05-28 181013](https://github.com/user-attachments/assets/2eb91262-9faf-44f9-b460-5d88d28dd4d9)

Info():

![Screenshot 2025-05-28 181023](https://github.com/user-attachments/assets/9d9e4547-bc39-4bb9-a9a3-15738c3689d8)

isnull().sum():

![Screenshot 2025-05-28 181033](https://github.com/user-attachments/assets/d8459a42-0d62-4f81-807d-bd34ae3839e9)


Prediction of y:

![Screenshot 2025-05-28 181049](https://github.com/user-attachments/assets/dadc7f2c-053b-4ce5-b5f8-224156e645b3)

Accuracy:

![Screenshot 2025-05-28 181056](https://github.com/user-attachments/assets/ad309ff2-4d7d-4e48-aab9-21481075d900)


Confusion Matrix:

![Screenshot 2025-05-28 181104](https://github.com/user-attachments/assets/342f489b-8493-4818-8668-5e6e25791301)


Classification Report:

![Screenshot 2025-05-28 181112](https://github.com/user-attachments/assets/b9d845a5-60ea-4688-8775-c82aed169682)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
