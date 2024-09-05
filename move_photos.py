###program that saves photos from database to image directories from openCv_facial_detection.py

import sqlite3
import numpy as np
import tensorflow as tf
import cv2

from tensorflow import keras
from PIL import Image
from io import BytesIO


def create_connection(database):

    conn = None
    try:
        conn = sqlite3.connect(database)
    except Error as e:
        print(e)
        
    return conn

def select_first(conn):
    
    cur = conn.cursor()
    cur.execute("SELECT frame FROM my_test")
    
    blob = cur.fetchall()
    
    return blob
        
connection = create_connection('faces_database.db')
blob = select_first(connection)

file_like = BytesIO(blob[0][0])
img = Image.open(file_like)



#change range to number of faces in faces database
for i in range(87):
    
    file_like = BytesIO(blob[i][0])
    img = Image.open(file_like)    
    #example, inserting pictures of phuc into his image directory
    string = f"C:/Users/spunk/Desktop/gittest/CSC450PROJ/phuc_nguyen/phuc_nguyen_{i+522:04d}.jpg"
    img.save(string, 'JPEG')

