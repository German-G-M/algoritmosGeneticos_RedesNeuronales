# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:01:14 2022

@author: germa
"""
# red neuronal con el data set IRIS (tiene 150 registros)
from sklearn import datasets
iris= datasets.load_iris() #importalmos desde scikitlearn
#print (iris)

#realizmos la división entre los "datos" y las "clases"
X= iris.data
y= iris.target
print ("datos iris:", X)
print("objetivos iris: ", y) # esto es la clase (0='setosa', 1='versicolor', 2='virginica' )

#*************dividimos los 150 registros en datos de "entrenamiento" y datos de "prueba"**********************
#en Scikit learn hay una opción que nos permite hacer la división
from sklearn.model_selection import train_test_split
#la librería "train_test_split" nos permite hacer la división de "train" y "test"
#train_test_split(X_original, y_original, tamaño_del_test)
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.25) #25% sera para la prueba "test"
print(y_test) #mostramos las pruebas

#************************ahora generamos el modelo************************************************************
#utilizamos una arquitectura de la red Neuronal llamada "MLP" (Multi Layer Percentron= perceptron multicapa)
# el perceptron es el modelo de una neurona simple
from sklearn.neural_network import MLPClassifier #importamos nuestro clasificador de red
model= MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
#ponemos tres capaz ocultas, cada una de 10 neuronas. Y 1000 iteraciones

#***********************Ahora entrenamos la red.**********************************************************
#Hay 2 formas de hacerlo: con la "fit" y la "fit transform". La "fit" es solo entrenar
print("inicio del entrenamiento...")
model.fit(X_train,y_train) #ingresan los datos de entrenamiento de "X" y de "y"
print("fin del entrenamiento")

#***********************ahora hacemos la predicción*********************************************************
prediccion = model.predict(X_test)#introducimos los datos de testeo
print ("datos de la prediccion del modelo: ",prediccion)
print("datos de prueba: ",y_test)

import matplotlib.pyplot as plt
plt.xlabel ("iteraciones")
plt.ylabel("magnitud de peridida")

'''
Este es el resultado de la corrida:
    
inicio del entrenamiento...
fin del entrenamiento
datos de la prediccion del modelo:  
    [2 1 1 2 1 1 0 0 2 1 2 1 1 1 1 2 0 0 0 0 1 2 0 1 2 0 1 1 0 0 0 2 0 2 1 2 2 0]
datos de prueba:  
    [2 1 1 2 1 1 0 0 2 1 2 1 1 1 1 2 0 0 0 0 1 2 0 1 2 0 1 1 0 0 0 2 0 2 1 2 2 0]

LA PREDICCIÓN DEL MODELO ES IDENTICA A LOS DATOS DE PRUEBA
'''

