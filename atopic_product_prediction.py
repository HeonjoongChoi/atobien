
#Common imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from pickle import dump
from pickle import load




#Import the data set

raw_data = pd.read_csv(r"E:\restapi_atobien\apps\model\Data\atopic_product.csv", header=0, encoding='CP949')
raw_data.columns
print("raw_data")
print(raw_data)
raw_data = raw_data[['Disease', 'Body_part', 'Product_Name']]
ip_addresses = raw_data.Product_Name.unique()
ip_dict_1 = dict(zip(ip_addresses, range(len(ip_addresses))))
print(ip_dict_1)
print("========================================================")
raw_data=raw_data.replace({'Product_Name':ip_dict_1})
ip_addresses = raw_data.Body_part.unique()
ip_dict_2 = dict(zip(ip_addresses, range(len(ip_addresses))))
print(ip_dict_2)
print("========================================================")
raw_data=raw_data.replace({'Body_part':ip_dict_2})
ip_addresses = raw_data.Disease.unique()
ip_dict_3 = dict(zip(ip_addresses, range(len(ip_addresses))))
print(ip_dict_3)
print("========================================================")
raw_data=raw_data.replace({'Disease':ip_dict_3})

#Import standardization functions from scikit-learn

from sklearn.preprocessing import StandardScaler

#Standardize the data set

scaler = StandardScaler()

scaler.fit(raw_data.drop("Product_Name", axis=1))

scaled_features = scaler.transform(raw_data.drop("Product_Name", axis=1))

scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop("Product_Name", axis=1).columns)

#Split the data set into training data and test data
print("raw_data")
print(raw_data)
from sklearn.model_selection import train_test_split

x = scaled_data
print(x)
print("========================================================")
y = raw_data["Product_Name"]

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.2)

#Train the model and make predictions

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 1)

model.fit(x_training_data, y_training_data)
# save the model
dump(model, open('atopic_product.pkl', 'wb'))


model = load(open('atopic_product.pkl', 'rb'))
predictions = model.predict(x_test_data)
print("x_test_data")
print(x_test_data)
print("========================================================")
print("predictions")
print(predictions)
print("========================================================")
product_name = list(ip_dict_1.keys())[list(ip_dict_1.values()).index(predictions[0])]
Disease_name = list(ip_dict_3.keys())[2]
Body_part = list(ip_dict_2.keys())[list(ip_dict_2.values()).index(raw_data['Body_part'][0])]
print("Disease Type:",Disease_name) 
print("Affected part of the body:",Body_part)
print("Recomended Product:",product_name) 





