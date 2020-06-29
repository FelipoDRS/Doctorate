# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:13:55 2020

@author: Felipo Soares
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os
path_image= r'E:\Downloads_HD\AnaliseImagem-ELISA\Analises_250620'

files= os.listdir(path_image)
for i in files:
    if '.jpg' in i:    
        image = cv2.imread(path_image+'\\'+i,1)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np.save(path_image+'\\'+i.replace('.jpg','.pkl'), image)

