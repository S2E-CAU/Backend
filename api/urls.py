#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 00:51:59 2021

@author: yoonseoha
"""

from django.urls import path, include
from .views import NumOfSolarCellAPIView

urlpatterns=[
    path("map", NumOfSolarCellAPIView.as_view())
]