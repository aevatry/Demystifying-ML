import numpy as np
import math
from mlxtend.image import extract_face_landmarks
import cv2
from scipy.spatial import distance
import pandas as pd

def getFrame(sec, vidcap):
    start = 100
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start + sec*1000)
    hasFrames,image = vidcap.read()
    return hasFrames, image


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = distance.euclidean(eye[1], eye[4])
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance.euclidean(eye[0], eye[1])
    p += distance.euclidean(eye[1], eye[2])
    p += distance.euclidean(eye[2], eye[3])
    p += distance.euclidean(eye[3], eye[4])
    p += distance.euclidean(eye[4], eye[5])
    p += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area /(p**2)
     

def mouth_over_eye(eye):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye

def normalization(series, ref_series):
    if type(series) != pd.Series:
        raise TypeError('Input is not a Pandas Series')
    mean = ref_series.mean()
    std = ref_series.std()
    series = (series - mean)/std
    return series

def appending(extracted:np.ndarray, output:np.ndarray, sequence_lenght: int, features:int):
    for i in range(0, len(extracted) - sequence_lenght +1):
        extrac_append = extracted[i:i+sequence_lenght].reshape((-1,sequence_lenght, features))
        output = np.append(output, extrac_append, axis = 0)
    return output

def relabel(labels:np.ndarray):
    new_labels = np.ndarray(shape = (labels.shape[0],1))
    for i in range(0,labels.shape[0]):
        new_labels[i] = labels[i][0]
    return new_labels

def VideoExtract(link: str):
    '''
    VideoExtract returns the 
    '''
    data = []
    vidcap = cv2.VideoCapture(link)
    sec = 0
    frameRate = 1
    success, image  = getFrame(sec, vidcap)
    count = 0
    while success and count < 240:
        landmarks = extract_face_landmarks(image)
        if sum(sum(landmarks)) != 0:
            count += 1
            data.append(landmarks)
            sec = sec + frameRate
            sec = round(sec, 2)
            success, image = getFrame(sec, vidcap)
            
        else:  
            sec = sec + frameRate
            sec = round(sec, 2)
            success, image = getFrame(sec, vidcap)
        
    data = np.array(data)
    features = []
    for d in data:
        eye = d[36:68]
        ear = eye_aspect_ratio(eye)
        mar = mouth_aspect_ratio(eye)
        cir = circularity(eye)
        mouth_eye = mouth_over_eye(eye)
        features.append([ear, mar, cir, mouth_eye])
    
    features = np.array(features).T
    df = pd.DataFrame(data = {'EAR' : features[0],'MAR' : features[1],'CIR' : features[2],'MOE' : features[3]})


    df_norm = df.copy()
    df_norm = df_norm.reindex(columns = df_norm.columns.tolist() + ['normEAR', 'normMAR', 'normCIR', 'normMOE'])


    for feature in ['EAR', 'MAR', 'CIR', 'MOE']:
        # need to rethink the normalization
        ref_series = ref_series[feature]
        
        series = series[feature]
        
        norm_series = normalization(series, ref_series)
        column_name = 'norm' + feature
        df_norm.iloc[norm_series.index, df_norm.columns.get_loc(column_name)] = norm_series
