#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:57:19 2019

Editado: viernes 29 de noviembre del 2019 

@author: Cësar Arturo Martinez Hernández
"""



#importar las bibliotecas necesarias 
import numpy as np #Bliotecas para operaciones matriciales y computo cientifico 
import imutils #Para operaciones rapids con imagenes
import cv2 #Vision computacional 
from imutils.object_detection import non_max_suppression #Para no mostrar un objeto multiples veces 
import matplotlib.pyplot as plt #Para mostrar el resultado de la imagenes 
import sys 
import serial as sr
import time

class detectorPersonas:
    
    
    
    """Esta es la version 5.0 de programa Clase_Detector_Personas.py 
    Esta version presenta una libreria que te permite detectar personas ya sea en una imagen
    o en un video
    
    En esta nueva version tendremos una nueva funcion que implementara la funcion
    traking para optimizar los tiempos, en comparacion a la funcion detectVideoFpF. Esta 
    nueva funcion o método es detectVideoTraking(). No es necesario asignar un nuevo path
    para usar esta función. Tampoco es necesario pasarle parametros. 
    
    Incluye un metodo setPathImg para asignar la ruta de la imagen que se quiere testear
    
    Para fines practicos,  se esta trabajando con una maquina entrenada. Por lo tanto,
    en esta version no es necesario pasarle el dataset al algoritmo. Esto será icluido 
    en las versiones posteriores
    
    Para mostrar los resulatdos del test se utliza la función detectImg para el objeto
    especifico que se haya instanceado 
    
    """
    
    
    def __init__(self, imgPath=None, videoPath=None):    
        self.imgPath=imgPath     #Direccion de la imagen
        self.videoPath=videoPath
        
        
    
    def setPathImg(self, imgPath): #Se asigna la ruta de la imagen que se va a testear 
        
        self.imgPath=imgPath
    
    def setPathVideo(self, videoPath):
        
        self.videoPath=videoPath
        
    def detectImg(self):
        
        #Haremos la deteccion de peatones haciendo uso de HOG 
        #Se inicializa descriptor HOG y SVM
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
        #indicamos la direccion de la imagen
        ruta_img=self.imgPath
        imagen=cv2.imread(ruta_img) #Funcion de opencv para cargar la imagen en memoria
        imagen= imutils.resize(imagen, width= min(400, imagen.shape[1])) #Funcion de imutils para reescalar la imagen a un tamaño adecuado
        
        #Detectar peatones en la imagen 
        (rectas, weights) =  hog.detectMultiScale(imagen, winStride=(4,4), padding =(8,8), scale=1.05)
        #(rectas, weights) es una tupla
        #La deteccion de objetos se lleva a cabo en la funcion .detectMultiScale(<Donde quieres encontrarlos>, <parametros que afectan el tiempo de ejecución>)
        # winStride: Es un parametro que dicta el tamaño de paso en el que se va desplazando una ventana dentro de la variable imagen en la direccion horizontal 
        #El parametro padding indica la zona fuera de la ventana winStride
        #EL parametro scale nos da una representacion piramidal de la imagen reescalada
        #Las caracteristicas hog se van axtraer para cada iagen rescalda y para cada ventana
        #El parametro padding nos indica el numero de pixeles que se rellenan dentro de la ventana deslizante antes de extraer las caracteristicas hog.
        #winStride =(2,2), (6,6), (8,8)
        #padding = (2,2), (4,4), (8,8)

        #para enmarcar a las personas
        rectas=np.array([[x,y,x+w,y+h] for (x,y,w,h) in rectas])
        #rectas = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rectas])
        #Para evitar que nos encierre con multiples cuadros un peaton es:
        eleccion = non_max_suppression(rectas, probs = None, overlapThresh=0.65)
        
        #dibujar en la imagen el rectangulo para los peatones 
        #Para esto necesitamos un ciclo 
        for (xA, yA, xB, yB) in eleccion:
            cv2.rectangle(imagen, (xA,yA), (xB,yB), (0,255,0), 2)
            
        
        #mostrar la imagen 

        imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)#Fucnion para invertir los canales y la imagen salga bien
        plt.imshow(imagen)
        
    def detectVideoFpF(self):
           
        #Se inicializa descriptor HOG y SVM
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
        
        #Inicializando captura de imagen desde archivo 
        cap = cv2.VideoCapture(self.videoPath)
        
        #ciclo que siempre este leyendo 
        while True:
            #Captura de imagenes
            #ruta_video=self.videoPath
            ret, imagen = cap.read() ##############################
            imagen=imutils.resize(imagen, width = 400)
            #Detectar peatones en la imagen 
            (rectas, weights)=hog.detectMultiScale(imagen, winStride = (4,4), padding = (8,8), scale = 1.05)
            
            #para enmarcar a las personas
            rectas=np.array([[x,y,x+w,y+h] for (x,y,w,h) in rectas])
            
            #Para evitar que nos encierre con multiples cuadros un peaton es:
            eleccion = non_max_suppression(rectas, probs = None, overlapThresh=0.65)
            
            #dibujar en la imagen el rectangulo para los peatones 
            #Para esto necesitamos un ciclo 
            for (xA, yA, xB, yB) in eleccion:
                cv2.rectangle(imagen, (xA,yA), (xB,yB), (0,255,0), 2)
            
            #Mostrar numero de peatones encontrados 
            if (len(eleccion)):
                print("{} peatones encontrados".format(len(eleccion)))
                print(len(eleccion))
                                #####COMUNICACIÓN SERIAL ###########
                
                puerto2=sr.Serial('/dev/ttyACM0', 9600) #asigna puerto 

                
                varP=len(eleccion)                    #variable a mandar es t
                
                
                print(puerto2.name)
                puerto2.write((str(varP)).encode('utf-8'))             #Mandar dato
                print(varP)
                print(type(varP))
                puerto2.close()
                
                
                
            cv2.imshow("imagen de salida", imagen)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    def detectVideoTraking(self):
        
        
        ##############ELECCIÓN DE TRAKER########
        
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TDL' ,'MEDIANFLOW', 'GOTURN', 'MOSSE' , 'CSRT']
        tracker_type = tracker_types[2]
    
  
        if tracker_type == 'BOOSTING':   
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
            
        
         #######Envez de definir una bouding box inicial, iniciamos el cpodigo 
        #que detecta personas################################################
        
        
        #Haremos la deteccion de peatones haciendo uso de HOG 
        #Se inicializa descriptor HOG y SVM
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        #Read video
        video = cv2.VideoCapture(self.videoPath)
    
        #Exit if video not opened 
        if not video.isOpened():
            
            print("Could not open video")
            sys.exit()
    
        #Read first frame 
        ok, frame = video.read()
        if not ok:
            print("Cannot read video file")
            sys.exit()
        

       
        #INICIALIZA EL HOG PARA LA IMAGEN
        found, _ = hog.detectMultiScale(frame)
        
        fig = plt.figure(figsize=(10, 6))   #Figura de este tamaño 
        ax = fig.add_subplot(111)           
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        from matplotlib import patches
        for f in found:                     #SE CREAN LAS BOXES QUE ENCIERRAN A LAS PRSONAS DETECTADAS 
            ax.add_patch(patches.Rectangle((f[0], f[1]), f[2], f[3], color='r', linewidth=3, fill=False))
        
        plt.savefig('detected.png')
        
   
        ok=tracker.init(frame,tuple(f))
        
        while(True):
            #Lee un nuevo frame 
            ok, frame= video.read()
            
            if not ok:
                break
            
            #Inicia el timmer 
            timer = cv2.getTickCount()
            
            #Descarga tracker 
            ok, bbox = tracker.update(frame)
            
            #Calcuar los frames por segundo
            fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer);
            
            #Draw bouding box
            if ok:
                #Tracking success 
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+ bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                
                
                #####COMUNICACIÓN SERIAL ###########
                
                puerto=sr.Serial('/dev/ttyUSB0', 9600) #asigna puerto 

                
                var=['t']                    #variable a mandar es t
                
                for i in range(len(var)):
                    print(puerto.name)
                    puerto.write(var[i].encode('utf-8'))             #Mandar dato
                    print(var[i])
                    puerto.close()
                
                
                
            
            
            
            else:
                #Traking Failure
                cv2.putText(frame, "Tracker failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                
                #puerto=sr.Serial('/dev/ttyUSB0', )
            #Display tracker type on frame 
            cv2.putText(frame,tracker_type + "Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        
            #Display FPS on frame
            cv2.putText(frame,"FPS:" + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        
            #Display result 
            cv2.imshow("Traking", frame)
        
            #Exit if ESC pressed
            k=cv2.waitKey(1) & 0xff
            if k==27 : break
        
       







        
objeto1=detectorPersonas()
objeto1.setPathVideo("")
#objeto1.setPathVideo(0)
objeto1.detectVideoTraking()    
#objeto1.detectVideoFpF()