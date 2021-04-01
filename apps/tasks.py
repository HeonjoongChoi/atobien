from __future__ import absolute_import, unicode_literals
from apps.models import CeleryScan
from apps.model.atopic_severity_level_prediction_appt import forecast
#from apps.model.atopic_severity_level_prediction_fixed import forecast

from celery import Task
from apps.celery import app
from rest_framework.response import Response
from rest_framework import status


from pickle import dump
from pickle import load

import os
from datetime import datetime 


import os.path

import pandas as pd


class ScanCode(Task):
    ignore_result = True
    #name = 'scancode'
    def __init__(self, *args, **kwargs):
        # scan_code_async 메소드에서 self 사용했으므로 초기화 할 필요없음 
        self.id_no = kwargs.get('id_no', None)
        

@app.task
def predict_code_async(consulting_date, appointment_date, disease_code, user_id, severity, id_no):    
 
    path = '/usr/src/app/Consultation_details/'
    dateFormatter = '%Y-%m-%d'
    Disease = ""
    severity_new = int(severity)
    print("============================")
    print(type(severity_new))
    print(type(consulting_date))
    print("============================")
    print(disease_code)
    print(type(disease_code))
    print("============================")
    print(type(user_id))
    print("============================")
    
    #Decision of Disease by Disease_code
    if  disease_code == '1' :
        Disease = 'Excoriation'
    if  disease_code == '2' :
        Disease = 'Erythema'
    if  disease_code == '3' :
        Disease = 'Lichenification'
    if  disease_code == '4' :
        Disease = 'Papulation'
   

    if os.path.isfile(path+user_id+'_'+Disease+'.csv') :
        
        #Read csv file
        raw_data = pd.read_csv(path+user_id+'_'+Disease+'.csv')
        raw_data.columns
        
        #Calculation of Time_duration
        time1 = datetime.strptime(consulting_date, dateFormatter)
        time2 = datetime.strptime(appointment_date, dateFormatter)
        date_duration = (time2.date()-time1.date()).days
        
        
            
        #Append Input in csv file 
        equip_series = pd.DataFrame({'Consulting_date':[consulting_date],
                                     'ID': [user_id],
                                     'Disease':[Disease],
                                     'Severity':[severity_new],
                                     'Time_duration': [date_duration], 
                                     'Disease_Code': [disease_code]},
                                    columns=['Consulting_date','ID', 'Disease','Severity','Time_duration', 'Disease_Code'])   
        raw_data = raw_data.append(equip_series, ignore_index=True)    
            
        #csv file save
        raw_data.to_csv(path+user_id+'_'+Disease+'.csv', mode='w', index=False)
                         
        #forecast the severity
        result = forecast(path+user_id+'_'+Disease+'.csv', Disease, consulting_date, appointment_date, severity)
        

        r = Response(result, status=status.HTTP_200_OK)
 
        if r.status_code == 200:
            return apply_predict_async.delay(result, id_no)
        else:
            return 'Some error has occured'
    
    else: 
                
        #Append Input in csv file 
        equip_series = pd.DataFrame({'Consulting_date':[consulting_date],
                                     'ID': [user_id],
                                     'Disease':[Disease],
                                     'Severity':[severity],
                                     'Time_duration': ['0'], 
                                     'Disease_Code': [disease_code]},
                                    columns=['Consulting_date','ID', 'Disease','Severity','Time_duration', 'Disease_Code'])
        equip_series.to_csv('/usr/src/app/Consultation_details/a.csv', index=False)
        os.rename('/usr/src/app/Consultation_details/a.csv', path+user_id+'_'+Disease+'.csv')
            
        
        def resultmodule() :
        
            return {
                "Current severity of the disease ": severity,
                "The Expected severity of the disease after one week ": severity,
                "The severity ": "The severity shall be worse until next treatment"
                }
        
        result = resultmodule()
        r = Response(result, status=status.HTTP_200_OK)
        if r.status_code == 200:
            return apply_predict_async.delay(result, id_no)
        else:
            return 'Some error has occured'
     




@app.task
def apply_predict_async(result, id_key):
    celery_scan = CeleryScan.objects.get(id_key=id_key)
    celery_scan.json_results = result
    celery_scan.is_complete = True
    celery_scan.save()