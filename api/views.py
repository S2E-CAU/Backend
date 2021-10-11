# Create your views here.
import json

from django.core.serializers.json import DjangoJSONEncoder
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import ImageSerializer

from django.http import HttpResponse
import base64
import os
import sys
import pandas as pd
from .process import run

sys.path.append(os.getcwd())

def convert2base64(img_dir):
    with open(img_dir, "rb") as image_file:
        base64str = base64.b64encode(image_file.read())
        return base64str


class NumOfSolarCellAPIView(APIView):
    def __init__(self):
        self.base_dir = os.getcwd()

    def post(self, request):
        # print(request.data)
        file_serializer = ImageSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            img_dir = file_serializer.data["file"]

            img_path = self.base_dir + img_dir
            out_dir = self.base_dir + "/media/solarcell.jpg"

            number = run(img_path, out_dir)
            img64 = convert2base64(out_dir).decode('utf-8')

            data = [
                {
                    'img': img64,
                    'number': number
                }
            ]

            json.dumps(data, cls=DjangoJSONEncoder)
            return HttpResponse(content=data, content_type='application/json')
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TestView(APIView):
    def __init__(self):
        self.base_dir = os.getcwd()

    def post(self, request):
        img_path = self.base_dir + "/media/image.jpg"
        out_dir = self.base_dir + "/media/solarcell.jpg"

        number = run(img_path, out_dir)
        img64 = convert2base64(out_dir).decode('utf-8')

        data = [
            {
                'img': img64,
                'number': number
            }
        ]

        json.dumps(data, cls=DjangoJSONEncoder)
        return HttpResponse(content=data, content_type='application/json')


class SolarPower(APIView):
    def __init__(self):
        self.base_dir = os.getcwd()

    def post(self, request):
        # print(request.data)

        csv = self.base_dir + "/media/solar.csv"
        csv_data = pd.read_csv(csv, sep=',')
        # csv_data.to_json(self.base_dir+"/media/solar.json", orient="records")

        with open(self.base_dir + "/media/solar.json", 'r') as f:
            json_data = json.load(f)
        data = json.dumps(json_data)
        return HttpResponse(content=data, content_type='application/json')
