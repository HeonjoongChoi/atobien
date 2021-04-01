
from rest_framework import serializers
from apps.models import FileModel
from apps.models import TextModel

class FileSerializer(serializers.ModelSerializer):
        
    class Meta:
        model = FileModel
        fields = '__all__'
        
class TextSerializer(serializers.ModelSerializer):
    
    class Meta:
        
        model = TextModel
        fields = '__all__'