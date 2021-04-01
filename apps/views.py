from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.http.request import QueryDict
from apps.serializers import FileSerializer, TextSerializer
from apps.models import CeleryScan
from apps.tasks import predict_code_async, apply_predict_async


from pickle import load

import os
from apps.utils import is_image
from apps import file_upload_path
from pytilhan.utils import log_util


import os.path
import numpy
import cv2

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from tensorflow.keras.preprocessing.image import img_to_array


class AtobienView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, req, *args, **kwargs):
        
        new_data = req.data.dict()
        
        # add new data to QueryDict(for mapping with Text, JSON)
        consulting_date = req.data['Consulting_Date']
        appointment_date = req.data['Appointment_Date']
        disease_code = req.data['Disease_Code']
        user_id = req.data['User_Id']
        severity = req.data['Severity']
    
        
        #add model about Celery
        celery_scan = CeleryScan(json_results='', is_complete=False)
        celery_scan.save()
        id_no = celery_scan.id_key
        
        new_data['category_id'] = consulting_date+'/'+appointment_date+'/'+disease_code+'/'+user_id+'/'+severity
        
        new_query_dict = QueryDict('', mutable=True)
        new_query_dict.update(new_data)
        text_serializer = TextSerializer(data = new_query_dict)
    
        if text_serializer.is_valid():
           
            text_serializer.save()
            print("save complete")
            print( text_serializer.data)
            disease_code = str(disease_code)
    
            print(type(severity))
            
            # create the object so scan can be applied
            result = predict_code_async.delay(consulting_date, appointment_date, disease_code, user_id, severity, id_no)
                    
            # return the response as HttpResponse
            url = 'localhost:8000/atobien_api_result/' + str(id_no)       
            return Response({"url" : url}, status=status.HTTP_200_OK)
        else:
            
            log_util.error(__name__ , text_serializer.errors)
            return Response(text_serializer.errors, status=status.HTTP_400_BAD_REQUEST)   
        
        
       
         
class SeverityResults(APIView):
    
    def get(self, req, *args, **kwargs):
        
        celery_scan = CeleryScan.objects.get(id_key=kwargs['pk'])
        print("celery_scan is complete???? ",celery_scan.is_complete)
        result = {"result": "in progress"  }

        if celery_scan.is_complete == True:

            result = celery_scan.json_results

        return Response(result, status=status.HTTP_200_OK)    
         


class AtobienView2(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, req, *args, **kwargs):
        model = tf.keras.models.load_model('/usr/src/app/atopic.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        # model = tf.keras.models.load_model(r'E:\restapi_atobien\apps\model\Model\atopic.h5',custom_objects={'KerasLayer':hub.KerasLayer})
        print("Done lodding")
        product = pd.read_csv('/usr/src/app/atopic_product.csv', header=0, encoding='CP949')
        # product = pd.read_csv(r'E:\restapi_atobien\atopic_product.csv', header=0, encoding='CP949')

        pd.set_option("display.precision", 8)

        # get requested data      
        new_data = req.data.dict()

        # requested file object
        file_name = req.data['file_name']

        # create full path for saving file        
        new_file_full_name = file_upload_path(file_name.name)

        # new file path
        file_path = '/'.join(new_file_full_name.split('/')[0:-1])

        # file extension
        file_ext = os.path.splitext(file_name.name)[1]

        # add new data to QueryDict(for mapping with ImageFile)
        new_data['file_ext'] = file_ext
        new_data['is_img'] = is_image(file_ext)
        new_data['file_path'] = file_path
        new_data['file_origin_name'] = req.data['file_name'].name
        new_data['file_save_name'] = req.data['file_name']

        # add new data to QueryDict(for mapping with Text)
        User_Name = req.data['User_Name']
        User_Sex = req.data['User_Sex']
        User_Age = req.data['User_Age']
        User_Skin_Type = req.data['User_Skin_Type']
        User_Acne_Freq = req.data['User_Acne_Freq']
        User_Skin_Resp = req.data['User_Skin_Resp']
        User_Skin_Stat = req.data['User_Skin_Stat']
        User_Affected_Part = req.data['User_Affected_Part']
        Symp_Score = req.data['Symp_Score']

        new_query_dict = QueryDict('', mutable=True)
        new_query_dict.update(new_data)

        file_serializer = FileSerializer(data=new_query_dict)

        if file_serializer.is_valid():

            file_serializer.save()

            # pre-process the image for classification
            image = cv2.imread(file_serializer.data['file_save_name'])
            image = cv2.resize(image, (224, 224))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            tf_model_predictions = model.predict(image)
            print("Prediction results shape:", tf_model_predictions)

            dataset_labels = np.array(
                ['Erythema1', 'Erythema2', 'Erythema3', 'Excoriation1', 'Excoriation2', 'Excoriation3',
                 'Lichenification1', 'Lichenification2', 'Lichenification3', 'Papulation1', 'Papulation2',
                 'Papulation3'])

            predicted_ids = np.argmax(tf_model_predictions, axis=-1)

            predicted_labels = dataset_labels[predicted_ids]

            raw_data = product[['Disease', 'Body_part', 'Product_Name']]
            ip_addresses = raw_data.Product_Name.unique()
            ip_dict_1 = dict(zip(ip_addresses, range(len(ip_addresses))))

            raw_data = raw_data.replace({'Product_Name': ip_dict_1})
            ip_addresses = raw_data.Body_part.unique()
            ip_dict_2 = dict(zip(ip_addresses, range(len(ip_addresses))))

            raw_data = raw_data.replace({'Body_part': ip_dict_2})
            ip_addresses = raw_data.Disease.unique()
            ip_dict_3 = dict(zip(ip_addresses, range(len(ip_addresses))))

            raw_data = raw_data.replace({'Disease': ip_dict_3})

            # Import standardization functions from scikit-learn

            from sklearn.preprocessing import StandardScaler

            # Standardize the data set

            scaler = StandardScaler()

            scaler.fit(raw_data.drop("Product_Name", axis=1))

            Affected_Part = -1;

            if User_Affected_Part == '0':
                Affected_Part = 0
                # df_product1 = df_product[df_product["Body_part"]== "Body"]
            if User_Affected_Part == '1':
                Affected_Part = 1
                # df_product1 = df_product[df_product["Body_part"]== "Face"]
            if User_Affected_Part == '2':
                Affected_Part = 2
                # df_product1 = df_product[df_product["Body_part"]== "Scalp"]
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
            Disease_no = -1
            if predicted_ids == [0] or predicted_ids == [1] or predicted_ids == [2]:
                Disease_no = 0
            if predicted_ids == [3] or predicted_ids == [4] or predicted_ids == [5]:
                Disease_no = 1
            if predicted_ids == [6] or predicted_ids == [7] or predicted_ids == [8]:
                Disease_no = 2
            if predicted_ids == [9] or predicted_ids == [10] or predicted_ids == [11]:
                Disease_no = 3

            data_series = pd.DataFrame({'Disease': [Disease_no], 'Body_part': [Affected_Part]},
                                       columns=['Disease', 'Body_part'])

            new_raw_data = raw_data.append(data_series, ignore_index=True)

            scaled_features = scaler.transform(new_raw_data.drop("Product_Name", axis=1))

            scaled_data = pd.DataFrame(scaled_features, columns=raw_data.drop("Product_Name", axis=1).columns)

            # Split the data set into training data and test data

            from sklearn.model_selection import train_test_split

            x = scaled_data

            model = load(open('/usr/src/app/atopic_product.pkl', 'rb'))
            predictions = model.predict(scaled_data)

            product_name = list(ip_dict_1.keys())[list(ip_dict_1.values()).index(predictions[-1])]

            df_product2 = product[product["Product_Name"] == product_name]

            def resultmodule():
                return {
                    "Name": User_Name,
                    "Age": User_Age,
                    "Disease name & severity(3steps): ": predicted_labels[0],
                    "Affected part of the body": df_product2["Body_part"].values[0],
                    "Recommended Product:": df_product2["Product_Name"].values[0],
                    "Product Type:": df_product2["Usage"].values[0],
                    "Link to buy the Product: ": df_product2["Link"].values[0]
                }

            results = resultmodule()

            return Response(results, status=status.HTTP_200_OK)

        else:

            log_util.error(__name__, file_serializer.errors)
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
