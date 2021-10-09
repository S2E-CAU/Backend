#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:24:26 2021

@author: yoonseoha
"""

import os
import cv2
from api.SemanticSegmentation.S2E_segmentation.full_pipeline import execute

def run(img_dir, save_dir):

    img = cv2.imread(img_dir)
    mask = cv2.imread(os.getcwd()+'/media/dongjak_mask.jpg')
        
    result, number =  execute(img, mask)
    
    cv2.imwrite(
            save_dir,
            result,
        )
    
    return number