#!/usr/bin/env python
# coding: utf-8

# In[4]:

import psycopg2
import psycopg2.extras

hostname = 'localhost'
database = 'postgres'
username = 'postgres'
pwd = 'test123'
port_id = 5432
conn = None

# import the rquired libraries.
from scipy.spatial import distance as dist
import numpy as np
import cv2
import dlib
from math import hypot
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
import threading
from imutils import face_utils
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from numpy import asarray
from Authenticpgm import findSimilarImage
from deepface import DeepFace
import csv
from datetime import datetime,date
import time
import psycopg2
import psycopg2.extras
TrueFlag = 'N'
font = cv2.FONT_HERSHEY_TRIPLEX

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20


RTC_CONFIGURATION = RTCConfiguration(
   # {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    {"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]}
    #rtc_configuration={"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]},
)
# Define the emotions.
emotion_labels = ['angry','attentive' ,'disgust', 'fear', 'happy','neutral','sad','sleepy', 'surprise','yawning']

# Load model.
classifier =load_model('emotion_detection_model_100epochs.h5')

# load weights into new model
#classifier.load_weights("model_weights_78.h5")

# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class VideoTransformer(VideoTransformerBase):
    
    def transform(self, frame):
        
        COUNTER = 0
        #process_change(img)
        img = frame.to_ndarray(format="bgr24")
        pic = frame.to_image()

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        #detector = dlib.get_frontal_face_detector()
        #predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        #faces = detector(img_gray)
        #landmark_list = predictor(img_gray, faces)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            
            #rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
            #landmark_list = predictor(img_gray, (x,y,w,h))
         
            #print('landmark ' , landmark_list)
            

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)

                

                #left_eye_ratio = eye_aspect_ratio([36, 37, 38, 39, 40, 41], landmark_list)
                #right_eye_ratio = eye_aspect_ratio([42, 43, 44, 45, 46, 47], landmark_list)
                #eye_open_ratio = (left_eye_ratio + right_eye_ratio) / 2
                #cv2.putText(img, str(eye_open_ratio), (0, 13), font, 0.5, (100, 100, 100))
                #print(left_eye_ratio,right_eye_ratio,eye_open_ratio)
                
                #inner_lip_ratio = mouth_aspect_ratio([60,62,64,66], landmark_list)
                #outter_lip_ratio = mouth_aspect_ratio([48,51,54,57], landmark_list)
                #mouth_open_ratio = (inner_lip_ratio + outter_lip_ratio) / 2
                #cv2.putText(img, str(mouth_open_ratio), (448, 13), font, 0.5, (100, 100, 100))
                #print(mouth_open_ratio)
                record = read_postgres('Ray')
               
                
                write_flag= 'N'
                if record['emotion'] != '' :
                    if (record['date1'] <= date.today() and
                     (int(round(time.time())) > int(record['timeepoch']) +15 )) : 
                        if output != record['emotion'] :
                        
                            write_flag = 'Y'
                else :
                    write_flag = 'Y'
                         
                if write_flag == 'Y' :
                    write_postgres('Ray', output)
                    
                
                #data_bhvr = read_postgres_bhvr('Ray')
                
                #if data_bhvr :
                #    savbhvr = data_bhvr['behavior']
                #else :
                #    savbhvr = ''
                #write_flag_bhvr = 'N'

                #if mouth_open_ratio > 0.380 and eye_open_ratio > 4.0 or eye_open_ratio > 4.30:
                #    bhvrstate = 'Sleepy'
                #else :
                #   bhvrstate ='not-sleepy'
                
                #if savbhvr != '' :
                #   if int(round(time.time())) > data_bhvr['timeepoch'] +15 : 
                #        if savbhvr != bhvrstate :
                #        
                #            write_flag_bhvr = 'Y'
                #else :
                #    write_flag_bhvr = 'Y'

                #if write_flag_bhvr == 'Y' :
                #   write_postgressbhvr('Ray',bhvrstate)

            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img
        
                
                  
            
            #label_position = (x, y-10)
            #cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
        #return img   


        #
        #return av.VideoFrame.from_image(pic)

class VideoTransformerbk(VideoTransformerBase):
    
    def transform(self, frame):
        
        img = frame.to_ndarray(format="bgr24")
        
        
       
        
       
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = MTCNN(min_face_size=15)
        faces = detector.detect_faces(img)
        
        for detected_face in faces :
            x, y, w, h = detected_face['box']
            
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 255), thickness=2)
            
            
            roi_gray = img_gray[y:y + h, x:x + w]
         
            
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
                
                data = read_postgres_bhvr('Ray')
                
               
                if data :
                    last_row = dat
                else :
                    last_row = ['06-11-2022', '123456789','00:00:01','', '']
                #print('file data' ,last_row[2], last_row[4])
                write_flag = 'N'
                if last_row[2] != '' :
                    if int(round(time.time())) > int(last_row[2]) + 15 : 
                        
                        if output != last_row[4] :
                           
                            write_flag = 'Y'
                else :
                        write_flag = 'Y'
                         
                if write_flag == 'Y' :

                    with open('emobehaviourtracker.csv', 'a' , newline = '') as file:
                                writer = csv.writer(file)
                           
                                writer.writerow([date.today(),int(round(time.time())), datetime.now().strftime("%H:%M:%S") ,output, "Not Applicable"])
                
                
                    

                
                    
            
            label_position = (x, y-10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
           
        return img
class VideoTransformer1(VideoTransformerBase):
    
    def recv(self, frame):
        #fig = plt.figure()
       
        img = frame.to_ndarray(format="bgr24")
        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = MTCNN()
        faces = detector.detect_faces(img_gray)
        #print(faces)
        for index,current_face_location in enumerate(faces):
            x, y, w, h = detected_face['box']
            (x, y) = (max(0, x), max(0, y))
            (x2, y2) = (min(im_w - 1, x+w), min(im_h - 1, y+h))
            face = img[y:y2, x:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x2, y2), color, 2)      
        return img
        
class FaceAuthenticator(VideoTransformerBase):
    
    def transform(self, frame):
        fig = plt.figure()
        #fig = plt.figure()
        img = frame.to_ndarray(format="bgr24")
        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        (im_h, im_w) = img.shape[:2]
        #image gray
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #detector = MTCNN()
        
        #faces = detector.detect_faces(img)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces =  detector(img_gray)
        for detected_face in faces :
            
            #x, y, w, h = detected_face['box']
            #(x, y) = (max(0, x), max(0, y))
            #(x2, y2) = (min(im_w - 1, x+w), min(im_h - 1, y+h))
            #face = img[y:y2, x:x2]
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            #face = cv2.resize(face, (224, 224))
            #face = img_to_array(face)
            #face = np.expand_dims(face, axis=0)
            #face = preprocess_input(face)
            color = (0, 255, 0)

            x,y = detected_face.left(), detected_face.top()
            x1,y1 = detected_face.right(), detected_face.bottom()
            cv2.rectangle(img, (x, y), (x1, y1), color, 2)  
            #cv2.rectangle(img, (x,y), (x2,y2), (0, 0, 255), 2)
            
            proper_face = 'Y'
            try:
                face_embedding = DeepFace.represent(img,  model_name = "VGG-Face")
            except :
                print('face is not captured properly')
                proper_face= 'N'
            if proper_face == 'Y' :
                imgname = findSimilarImage(face_embedding) 
                
            else:
                imgname='Face_not_captured'
                print(proper_face)
            #cv2.putText(img, imgname,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, imgname, (x, y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))

                              
        return img
def import_csv(csvfilename):
    data = []
    row_index= 0
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=',')
        for row in reader:
            if row:
                row_index += 1
                columns = [str(row_index), row[0], row[1], row[2],row[3]]
                data.append(columns)
    return data
def mid(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def process_change(img) :

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #detector = MTCNN(min_face_size=15)
        #faces = detector.detect_faces(img)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces = detector(img_gray)
        
        for detected_face in faces :
            #x, y, w, h = detected_face['box']
            x,y = detected_face.left(), detected_face.top()
            x1,y1 = detected_face.right(), detected_face.bottom()
            #cv2.rectangle(img=img, pt1=(x, y), pt2=(
            #    x + w, y + h), color=(0, 255, 255), thickness=2)
            #cv2.rectangle(img=img, pt1=(x, y), pt2=(
            #   x1, y1 ), color=(0, 255, 255), thickness=2)
            
            #roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = img_gray[y:y1 , x:x1]
            
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
         
            if np.sum([roi_gray]) != 0:
             
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                landmark_list = predictor(img_gray, detected_face)

                left_eye_ratio = eye_aspect_ratio([36, 37, 38, 39, 40, 41], landmark_list)
                right_eye_ratio = eye_aspect_ratio([42, 43, 44, 45, 46, 47], landmark_list)
                eye_open_ratio = (left_eye_ratio + right_eye_ratio) / 2
                #cv2.putText(img, str(eye_open_ratio), (0, 13), font, 0.5, (100, 100, 100))
                print(left_eye_ratio,right_eye_ratio,eye_open_ratio)
                
                inner_lip_ratio = mouth_aspect_ratio([60,62,64,66], landmark_list)
                outter_lip_ratio = mouth_aspect_ratio([48,51,54,57], landmark_list)
                mouth_open_ratio = (inner_lip_ratio + outter_lip_ratio) / 2;
                #cv2.putText(img, str(mouth_open_ratio), (448, 13), font, 0.5, (100, 100, 100))

                

                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
                #time_start = time.time()
                #print(savoutput)
                #data = import_csv('emobehaviourtracker.csv')

                record = read_postgres('Ray')
               
                
                write_flag= 'N'
                if record['emotion'] != '' :
                    if (record['date1'] <= date.today() and
                     (int(round(time.time())) > int(record['timeepoch']) +15 )) : 
                        if output != record['emotion'] :
                        
                            write_flag = 'Y'
                else :
                    write_flag = 'Y'
                         
                if write_flag == 'Y' :
                    write_postgres('Ray', output)
                    
                
                data_bhvr = read_postgres_bhvr('Ray')
                
                if data_bhvr :
                    savbhvr = data_bhvr['behavior']
                else :
                    savbhvr = ''
                write_flag_bhvr = 'N'

                if mouth_open_ratio > 0.380 and eye_open_ratio > 4.0 or eye_open_ratio > 4.30:
                    bhvrstate = 'Sleepy'
                else :
                    bhvrstate ='not-sleepy'
                
                if savbhvr != '' :
                    if int(round(time.time())) > data_bhvr['timeepoch'] +15 : 
                        if savbhvr != bhvrstate :
                        
                            write_flag_bhvr = 'Y'
                else :
                    write_flag_bhvr = 'Y'

                if write_flag_bhvr == 'Y' :
                   write_postgressbhvr('Ray',bhvrstate)
        return
   

def eye_aspect_ratio(eye_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(eye_landmark[0]).x, face_roi_landmark.part(eye_landmark[0]).y)
    right_point = (face_roi_landmark.part(eye_landmark[3]).x, face_roi_landmark.part(eye_landmark[3]).y)

    center_top = mid(face_roi_landmark.part(eye_landmark[1]), face_roi_landmark.part(eye_landmark[2]))
    center_bottom = mid(face_roi_landmark.part(eye_landmark[5]), face_roi_landmark.part(eye_landmark[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

def mouth_aspect_ratio(lips_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(lips_landmark[0]).x, face_roi_landmark.part(lips_landmark[0]).y)
    right_point = (face_roi_landmark.part(lips_landmark[2]).x, face_roi_landmark.part(lips_landmark[2]).y)

    center_top = (face_roi_landmark.part(lips_landmark[1]).x, face_roi_landmark.part(lips_landmark[1]).y)
    center_bottom = (face_roi_landmark.part(lips_landmark[3]).x, face_roi_landmark.part(lips_landmark[3]).y)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if hor_line_length == 0:
        return ver_line_length
    ratio = ver_line_length / hor_line_length
    return ratio

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance
def read_postgres(userid):
    record = ''
    try:
        with psycopg2.connect(
                host = hostname,
                dbname = database,
                user = username,
                password = pwd,
                port = port_id) as conn:
        
                    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                        cur.execute("SELECT * FROM emotiontracker where userid = 'Ray' ORDER BY timeepoch DESC LIMIT 1")
                        for record in cur.fetchall():
                            pass
               
    except Exception as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return record

def write_postgressbhvr(userid,bhvrstate) :

    try:
        with psycopg2.connect(
                        host = hostname,
                        dbname = database,
                        user = username,
                        password = pwd,
                        port = port_id) as conn:
                         with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                              insert_script  = 'INSERT INTO behaviortracker ( userid, date1, timeepoch , time, behavior) VALUES (%s, %s,%s, %s,%s)'
                              insert_values = [(userid,date.today(), round(time.time()), datetime.now().strftime("%H:%M:%S"),bhvrstate)]
                              for record in insert_values:
                                     cur.execute(insert_script, record)
    except Exception as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return

def read_postgres_bhvr(userid):
    record = ''
    try:
        with psycopg2.connect(
                host = hostname,
                dbname = database,
                user = username,
                password = pwd,
                port = port_id) as conn:
        
                    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                        cur.execute("SELECT * FROM behaviortracker where userid = 'Ray' ORDER BY timeepoch DESC LIMIT 1")
                        for record in cur.fetchall():
                            pass
               
    except Exception as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return record
def write_postgres(userid,emotionstate):
    try:
        with psycopg2.connect(
                        host = hostname,
                        dbname = database,
                        user = username,
                        password = pwd,
                        port = port_id) as conn:
                         with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                              insert_script  = 'INSERT INTO emotiontracker ( userid, date1, timeepoch , time, emotion) VALUES (%s, %s,%s, %s,%s)'
                              insert_values = [(userid,date.today(), round(time.time()), datetime.now().strftime("%H:%M:%S"), emotionstate)]
                              for record in insert_values:
                                     cur.execute(insert_script, record)
    except Exception as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return

def main():
    # Face Analysis Application #
    st.title("AI Tutor -Student Monitoring ")
    activiteis = ["Home", "Registration", "Sign In" , 'Face Authentication','Start Class']
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.image("thirdeye.jfif", use_column_width=True)
    

    # Homepage.
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#FC4C02;padding:0.5px">
                             <h4 style="color:white;text-align:center;">
                            AI Tutor -Student Monitoring.
                             </h4>
                             </div>
                             </br>"""

        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
        * AiTutor is an AI based learning management system
        * AiTutor that will be able to do Facial Recognition and Behavior Analytics
        * Please choose options at the left hand corner to navigate further.
                 """)

    # Live Face Emotion Detection.
    elif choice == "Registration": 
        st.header("Registration")
        st.subheader('''
        Please enter necessary details to proceed further!!!
         
        ''')
        with st.form("my_form"):
        
         mailid = st.text_input("Please enter mailid")
         Firstname =   st.text_input("Please enter First Name")
         Lastname  =    st.text_input("Please enter Last Name")
         Password  =    st.text_input("Please enter password")
         confPassword = st.text_input("Please confirm password")

         submitted = st.form_submit_button("Submit")
         
       

    # About.
    elif choice == "Sign In":
        st.header("Sign In")
        st.subheader('''
        Please Sign In and authenicate yourself for Accessing Live Session!!!
         
        ''')
        with st.form("my_form"):
         mailid = st.text_input("Please enter mailid")
         
         password = st.text_input("Please enter password")
         
         FaceAuthenticate = st.form_submit_button("Click for Face Authentication")
         #print('before' , FaceAuthenticate)
        
         if FaceAuthenticate:
                 
            st.write(f'hello {mailid}')
            webrtc_streamer(key="example", video_processor_factory=VideoTransformer)
            print('after' , FaceAuthenticate)
    elif choice == "Start Class":
        st.header("Webcam Live Feed")
        st.subheader('''
        Welcome to the other side of the SCREEN!!!
        * Your facial expressions will be analyzed and displayed in the screen. 
        ''')
        st.write("1. Click Start to open your camera and give permission for prediction")
        st.write("2. This will predict your emotion.")
        st.write("3. When you done, click stop to end.")
        
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer,rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        })
        
    elif choice == "Face Authentication":
        st.header("Webcam Live Feed")
        st.subheader('''
        Welcome to the other side of the SCREEN!!!
        * You will be authenticated through your facial features. 
        ''')
        st.write("1. Click Start to open your camera and give permission for prediction")
        
        st.write("2. This will authenticate you through capturing your face.")
        st.write("3. When you done, click stop to end.")
        webrtc_streamer(key="example1", video_processor_factory=FaceAuthenticator)
    else:
        pass


if __name__ == "__main__":
    main()







