
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from shutil import copy2
import csv
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import pandas as pd
from keras.models import load_model
plt.switch_backend('agg')
import cv2
import imutils
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from pickle import dump
from pickle import load

#model = load_model("atopic.h5")
model = tf.keras.models.load_model(r'E:\restapi_atobien\atopic.h5',custom_objects={'KerasLayer':hub.KerasLayer})
print("Done lodding")
product = pd.read_csv(r"E:\restapi_atobien\atopic_product.csv", header=0, encoding='CP949')
pd.set_option("display.precision", 8)



test_root =r"E:\restapi_atobien\pa2.png"
# load the image
image = cv2.imread(test_root)
output = imutils.resize(image, width=400)

# pre-process the image for classification
image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

tf_model_predictions = model.predict(image)
print("Prediction results shape:", tf_model_predictions)

dataset_labels = np.array(['Erythema1', 'Erythema2', 'Erythema3', 'Excoriation1', 'Excoriation2', 'Excoriation3', 'Lichenification1', 'Lichenification2', 'Lichenification3', 'Papulation1', 'Papulation2', 'Papulation3'])
print(dataset_labels)
print("========================================================")
predicted_ids = np.argmax(tf_model_predictions, axis=-1)
print("predicted_ids")
print(predicted_ids)
predicted_labels = dataset_labels[predicted_ids]
print("========================================================")  
print("predicted_ids")
print(predicted_ids)
print("========================================================")  

            
#df_product = df_product[df_product["Body_part"]== "Face"]


 

raw_data = product[['Disease', 'Body_part', 'Product_Name']]


ip_addresses = raw_data.Product_Name.unique()
ip_dict_1 = dict(zip(ip_addresses, range(len(ip_addresses))))
#print(ip_dict_1)
#print("========================================================")
raw_data=raw_data.replace({'Product_Name':ip_dict_1})
ip_addresses = raw_data.Body_part.unique()
ip_dict_2 = dict(zip(ip_addresses, range(len(ip_addresses))))
#print(ip_dict_2)
#print("========================================================")
raw_data=raw_data.replace({'Body_part':ip_dict_2})
ip_addresses = raw_data.Disease.unique()
ip_dict_3 = dict(zip(ip_addresses, range(len(ip_addresses))))
#print(ip_dict_3)
#print("========================================================")
raw_data=raw_data.replace({'Disease':ip_dict_3})

#Import standardization functions from scikit-learn

from sklearn.preprocessing import StandardScaler

#Standardize the data set

scaler = StandardScaler()
print("========================================================")  
print("raw_data")
print(raw_data)
print("========================================================")
scaler.fit(raw_data.drop("Product_Name", axis=1))
print("========================================================")  
print("raw_data")
print(raw_data)
print("========================================================")
User_Affected_Part = 1

if User_Affected_Part== 0:
    Affected_Part=0
    #df_product1 = df_product[df_product["Body_part"]== "Body"]
if User_Affected_Part== 1:
    Affected_Part=1    
    #df_product1 = df_product[df_product["Body_part"]== "Face"]
if User_Affected_Part== 2:
    Affected_Part=2
    #df_product1 = df_product[df_product["Body_part"]== "Scalp"]        
"""
if str(predicted_labels[0][:-1])=='Erythema':
    Disease_no=0
if str(predicted_labels[0][:-1])=='Excoriation':
    Disease_no=1
if str(predicted_labels[0][:-1])=='Lichenification':
    Disease_no=2
if str(predicted_labels[0][:-1])=='Papulation':
    Disease_no=3
"""    
Disease_no=-1
if predicted_ids==[0] or predicted_ids==[1] or predicted_ids==[2]:
    Disease_no=0
if predicted_ids==[3] or predicted_ids==[4] or predicted_ids==[5]:
    Disease_no=1
if predicted_ids==[6] or predicted_ids==[7] or predicted_ids==[8]:
    Disease_no=2
if predicted_ids==[9] or predicted_ids==[10] or predicted_ids==[11]:
    Disease_no=3    
    
data_series = pd.DataFrame({'Disease': [Disease_no], 'Body_part': [Affected_Part]}, columns=['Disease', 'Body_part'])

new_raw_data = raw_data.append(data_series,ignore_index=True)
print("========================================================")  
print("data_series")
print(data_series)
print("========================================================")
scaled_features = scaler.transform(new_raw_data.drop("Product_Name", axis=1))

print("========================================================")  
print("scaled_features")
print(scaled_features)
print("========================================================")

scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop("Product_Name", axis=1).columns)

print("========================================================")
#Split the data set into training data and test data

from sklearn.model_selection import train_test_split

x = scaled_data
print("========================================================")
print("scaled_data")
print(scaled_data.tail(1))
print("========================================================")

#x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.2)

#Train the model and make predictions
model = load(open('atopic_product.pkl', 'rb'))

predictions = model.predict(scaled_data.tail(1))

#print("x_test_data")
#print(x_test_data)
print("========================================================")
print("predictions")
print(predictions)
print("========================================================")
#Disease_name = list(ip_dict_3.keys())[0]
#print("========================================================")
#print(predicted_labels[0])
#print("========================================================")
product_name = list(ip_dict_1.keys())[list(ip_dict_1.values()).index(predictions[-1])]
#Body_part = list(ip_dict_2.keys())[list(ip_dict_2.values()).index(raw_data['Body_part'][0])]

if Affected_Part==0:
    part_of_body = 'Body'
if Affected_Part==1:
    part_of_body = 'Face'
if Affected_Part==2:
    part_of_body = 'Scalp'
    
print(product_name)
#print(df_product1)
df_product2 = product[product["Product_Name"]==product_name]
print(df_product2)
print("Affected part of the body:",part_of_body)
print("Disease name & severity(3steps) : ",predicted_labels[0])
print("Recommended Product:",df_product2.head(1)["Product_Name"].values[-1])
print("Product Type:",df_product2.head(1)["Usage"].values[-1])
print("Link to buy the Product: ", df_product2.head(1)["Link"].values[-1])
   

