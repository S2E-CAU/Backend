#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 01:13:18 2021

@author: yoonseoha
"""

from rest_framework import serializers
from .models import mapImage

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = mapImage
        fields = "__all__"