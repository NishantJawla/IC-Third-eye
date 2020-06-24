#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:05:58 2019
by Nishant Jawla

"""


import cv2
import os
import numpy as np
import shutil
import serial
#ser =serial.Serial('/dev/ttyACM0',9600,timeout=1)

mouth_cascade=cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
nose_cascade=cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_mcs_eyepair_big.xml")
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

if nose_cascade.empty():
    print("error")

if nose_cascade.empty():
    print("error")
    
cap=cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
ret, frame = cap.read()
fgmask = fgbg.apply(frame)


ds_factor=0.6

kernel = np.ones((5,5), np.uint8) 
#out = cv2.VideoWriter(
'fr.avi',
#cv2.VideoWriter_fourcc(*'MJPG'),
#15.,
#(384,288))
print("ayush")
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    mouth_rects=mouth_cascade.detectMultiScale(gray,1.3,5)
    m=np.array(mouth_rects)
    
    for (x,y,w,h) in mouth_rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        break
    face_rects=face_cascade.detectMultiScale(gray,1.3,5)
    f=np.array(face_rects)    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        break
    nose_rects=nose_cascade.detectMultiScale(gray,1.3,5)
    n=np.array(nose_rects)    
    for (x,y,w,h) in nose_rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        break
    eye_rects=eye_cascade.detectMultiScale(gray,1.3,5)
    e=np.array(eye_rects)    
    for (x,y,w,h) in eye_rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        break
    #out.write(frame.astype('uint8')) 
    if f.size !=0 :
        if n.size==0 and m.size==0:
             cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'H')
        if n.size ==0 and m.size==0 and e.size!=0:
            cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
            #ser.write(b'H')        
        if n.size ==0 and e.size==0 and m.size==0:
            cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
            #ser.write(b'H')
        if e.size==0 and m.size==0:
             cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'H')
        if e.size==0 and n.size==0:
             cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'H')             
        if n.size!=0 and e.size!=0:
             cv2.putText(frame,"not mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'L')
        if e.size!=0 and m.size!=0:
             cv2.putText(frame,"not mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
        if n.size!=0 and m.size!=0:
             cv2.putText(frame,"not mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'L')
    
    if f.size==0:
        if n.size==0 and m.size==0:
             cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'H')
        if n.size!=0 and e.size!=0 and m.size==0:    
             cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'H')
        if n.size ==0 and m.size==0 and e.size!=0:
            cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
            #ser.write(b'H')        
        if e.size==0 and m.size==0:
             cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'H')
        if e.size==0 and n.size==0:
             cv2.putText(frame,"mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'H')
        if e.size==0 and m.size!=0 and n.size!=0:
             cv2.putText(frame,"not mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'L')
        if e.size!=0 and n.size!=0 and m.size!=0:
             cv2.putText(frame,"no mask",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'L')
        if n.size==0 and e.size==0 and m.size==0:
             cv2.putText(frame,"no face",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_TRIPLEX, 0.8,  (0,0,255), 1)
             #ser.write(b'L')
    
#    start_timer=time.time()
#    result=time.localtime(start_timer)
#    compare=5
#    cv2.putText(frame,"Count:{time}".format(time=result.tm_sec),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
#    if result.tm_sec==compare:
#        cv2.putText(frame,"ALERT",(60,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,25
    
    #cv2.imshow('Detector',frame)
    c=cv2.waitKey(1)
    if c==27:
        break
    elif c==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#source='/home/anmol/Desktop/Third-Eye/fr.avi'
#des='/home/anmol/Desktop/Third-Eye/data/fr.avi'
#shutil.copyfile(source, des)
#os.remove('fr.avi')