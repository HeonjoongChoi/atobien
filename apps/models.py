from django.db import models
from apps import file_upload_path_for_db
# from uploadapp import file_upload_path


    
class FileModel(models.Model):
    
    # 실제 디스크에 저장되는 파일 절대 경로
    file_save_name  = models.FileField      ( upload_to=file_upload_path_for_db,  blank=False, null=False)
    # 파일의 원래 이름
    file_origin_name= models.CharField      ( max_length=100)    
    # 파일 저장 경로
    file_path       = models.CharField      ( max_length=100)
    # 파일 생성일
    create_date     = models.DateTimeField  ( auto_now_add = True)
    # 파일 확장자
    file_ext        = models.CharField      ( max_length=10)
    # 이미지 여부
    is_img          = models.BooleanField   ( default=False)
    
    def __str__(self):
        return self.file_origin_name
    
    class Meta:
        ordering = ['create_date']      
        #db_table = 'file_box'
        
class TextModel(models.Model):
          
        category_id = models.CharField(max_length=100)
        create_date = models.DateTimeField(auto_now_add=True)
     
     #User_Age = models.TextField(max_length=1000)
     #original_text = models.TextField(max_length=1000)
     #new_text = models.TextField(max_length=1000)
     #original_text_length = models.IntegerField()
     #new_text_length = models.IntegerField()
     #create_date = models.DateTimeField(auto_now_add=True)

     
# Celery Model
class CeleryScan(models.Model):
    id_key = models.AutoField(primary_key = True)
    json_results = models.CharField(max_length = 20000, null=True, blank=True)
    is_complete = models.BooleanField(default = False)

    def __str__(self):
        return str(self.id_key)     
    