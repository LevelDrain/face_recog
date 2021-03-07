# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from facenet_pytorch import MTCNN 
from PIL import Image, ImageDraw

img_path="test.png"
img=Image.open(img_path)
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    min_face_size=20, 
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=True, 
    select_largest=True, 
    keep_all=False, 
    device=None)
face=mtcnn(img, save_path="output.png")

boxes,props,points=mtcnn.detect(img,landmarks=True)
img_draw=img.copy()
draw=ImageDraw.Draw(img_draw)
draw.rectangle(boxes[0].tolist(),width=5)
img_draw.save('out2.png')