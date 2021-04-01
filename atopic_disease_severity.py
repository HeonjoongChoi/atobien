#Common imports

import numpy as np
from datetime import datetime 
import os 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from pickle import dump
from pickle import load

from apps.model.atopic_severity_level_prediction_appt import forecast

#Import the data set


        


path = 'E:/restapi_atobien/Consultation_details/'
input_id = 'choi'
input_date = '03/24/2021'
consulting_date = '2018-08-14'
appointment_date = '2018-08-21' 
input_disease_code = '3'
dateFormatter = '%m/%d/%Y'
Disease = ""

if os.path.isfile(path+input_id+'.csv') :
    
    #Read csv file
    raw_data = pd.read_csv(path+input_id+'.csv')
    raw_data.columns
    
    #Calculation of Time_duration
    time1 = datetime.strptime(input_date, dateFormatter)
    time2 = datetime.strptime((str(raw_data['Consulting_date'].values[-1])), dateFormatter)
    time_duration = (time1.date()-time2.date()).days
    
    #Decision of Disease by Disease_code
    if input_disease_code == '1' :
        Disease = 'Excoriation'
    if  input_disease_code == '2' :
        Disease = 'Erythema'
    if  input_disease_code == '3' :
        Disease = 'Lichenification'
    if  input_disease_code == '4' :
        Disease = 'Papulation' 
    
    
    
    a = forecast('E:/restapi_atobien/Consultation_details/Consultation_details.csv', Disease,consulting_date,appointment_date)
    print(a)
    #Append Input in csv file 
    equip_series = pd.DataFrame({'Consulting_date':[input_date],
                                 'ID': [input_id],
                                 'Disease':[Disease],
                                 'Severity':[1],
                                 'Time_duration': [time_duration], 
                                 'Disease_Code': [input_disease_code]},
                                columns=['Consulting_date','ID', 'Disease','Severity','Time_duration', 'Disease_Code'])   
    raw_data = raw_data.append(equip_series, ignore_index=True)
    
    #csv file save
    raw_data.to_csv(path+input_id+'.csv', mode='w', index=False)
    
    #Read csv file for index_col = 0
    #new_raw_data = pd.read_csv(path+input_id+'.csv', index_col = 0)
    new_raw_data = pd.read_csv(path+input_id+'.csv', index_col = 0)
    new_raw_data.columns
    new_raw_data = new_raw_data[['Severity', 'Time_duration', 'Disease_Code']]
    print("====================================================")
    print("new_raw_data")
    print(new_raw_data)
    print("====================================================")
    
    
    #Import standardization functions from scikit-learn
    from sklearn.preprocessing import StandardScaler
    
    #Standardize the data set
    scaler = StandardScaler()
    
    scaler.fit(new_raw_data.drop("Severity", axis=1))
    
    scaled_features = scaler.transform(new_raw_data.drop("Severity", axis=1))    
    print("====================================================")
    print("scaled_features")
    print(scaled_features)
    print("====================================================")
    
    scaled_data = pd.DataFrame(scaled_features, columns = new_raw_data.drop("Severity", axis=1).columns)    
    print("====================================================")
    print("scaled_data")
    print(scaled_data)
    print("====================================================")
    
    
    #Split the data set into training data and test data    
    from sklearn.model_selection import train_test_split    
    x = scaled_data
    y = new_raw_data["Severity"]    
    x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.2)
   
    print("====================================================")
    print("x_test_data")
    print(x_test_data)
    print("====================================================")
    
    print("====================================================")
    print("y_training_data")
    print(y_training_data)
    print("====================================================")
    
    #Train the model and make predictions
    from sklearn.neighbors import KNeighborsClassifier   
    model = KNeighborsClassifier(n_neighbors = 1)   
    model.fit(x_training_data, y_training_data)
    
    # save the model
    dump(model, open('atopic_severity.pkl', 'wb'))
    model = load(open('atopic_severity.pkl', 'rb'))
    predictions = model.predict(x_test_data) 
    print("====================================================")
    print("predictions")
    print(predictions[-1])
    print("====================================================")
    
    print("====================================================")
    print("y_test_data")
    print(y_test_data[-1])
    print("====================================================")
    
    #Save the severity in csv file
    raw_data = pd.read_csv(path+input_id+'.csv')
    raw_data['Severity'].values[-1] = y_test_data[-1]
    raw_data.to_csv('E:/restapi_atobien/Consultation_details/'+input_id+'.csv', mode='w', index=False)
    print(raw_data)
    
    print ("Curent severity of the disease:", y_test_data[-1])
    print ("Severity of the Disease during the next appointment :", predictions[-1])
    if(y_test_data[-1] > predictions[-1]):
            print("The severity shall be better until next treatment")
    else:
        print("The severity shall be worse until next treatment")
        
else :
    time_duration = 0
    severity = 1   
    
    if  input_disease_code == '1' :
        Disease = 'Excoriation'
    if  input_disease_code == '2' :
        Disease = 'Erythema'
    if  input_disease_code == '3' :
        Disease = 'Lichenification'
    if  input_disease_code == '4' :
        Disease = 'Papulation'    
    now = datetime.now()
    now_date = now.strftime('%m/%d/%Y')
    dataframe = pd.DataFrame({'Consulting_date':[now_date],
                                 'ID': [input_id],
                                 'Disease':[Disease],
                                 'Severity':[severity],
                                 'Time_duration': [time_duration], 
                                 'Disease_Code': [input_disease_code]},
                                columns=['Consulting_date','ID', 'Disease','Severity','Time_duration', 'Disease_Code'])
    dataframe.to_csv(r'E:/restapi_atobien/Consultation_details/a.csv', index=False)
    os.rename('E:/restapi_atobien/Consultation_details/a.csv', 'E:/restapi_atobien/Consultation_details/'+input_id+'.csv')
    
    
    new_raw_data = pd.read_csv(path+input_id+'.csv', index_col = 0)
    new_raw_data.columns
    new_raw_data = new_raw_data[['Severity', 'Time_duration', 'Disease_Code']]
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    scaler.fit(new_raw_data.drop("Severity", axis=1))
    
    scaled_features = scaler.transform(new_raw_data.drop("Severity", axis=1))    
    
    scaled_data = pd.DataFrame(scaled_features, columns = new_raw_data.drop("Severity", axis=1).columns)
    
    from sklearn.model_selection import train_test_split    
    x = scaled_data
    y = new_raw_data["Severity"]    
    #x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.1, train_size= 0.9)
    
    from sklearn.neighbors import KNeighborsClassifier   
    model = KNeighborsClassifier(n_neighbors = 1)   
    model.fit(x, y)
    
    dump(model, open('atopic_severity.pkl', 'wb'))
    model = load(open('atopic_severity.pkl', 'rb'))
    predictions = model.predict(x)
    
    dataframe = pd.read_csv(path+input_id+'.csv')
    dataframe['Severity'].values[-1] = y[-1]
    dataframe.to_csv('E:/restapi_atobien/Consultation_details/'+input_id+'.csv', mode='w', index=False)
   
    
    print ("Curent severity of the disease:", y)
    print ("Severity of the Disease during the next appointment :", predictions[-1])
    if(y[-1] > predictions[-1]):
            print("The severity shall be better until next treatment")
    else:
        print("The severity shall be worse until next treatment")
    
   