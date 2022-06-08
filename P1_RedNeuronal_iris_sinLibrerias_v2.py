# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:56:58 2022

@author: germa
"""

# red neuronal con el data set iris sin el uso de librerías

#****************prepaamos el dataset********************

from sklearn import datasets
iris =datasets.load_iris() #cargamos el iris de sde scikitlearn
#dividimos los datos de la clase
X =iris.data #4 columnas de 150 registos
y= iris.target #columna clase de 150registros
print ("Datos",X)
print ("Clase",y)

#por conveniencia convertiremos la clasificación
#a un problema binario cambiando la clase del iris
# ponemos 1 por "setosa" y 0 por "no es setosa"
import numpy as np
y_modif= np.array([1 if i==0 else 0 for i in y]) #cambiamos setosa=1, no_setosa = 0
print ("clase modoficada: ", y_modif)

#**************construimos nuestra RED NEURONAL********************
#La capa de entrada tiene 4 neuronas (columnas). Las caracteristicas de "iris": 1)sepal_leght 2)sepal_width 3)petal_leght 4)petal_width
#la capa de salida tiene solo una neurona, por que lo convertimos a un problema de casificación binario: setosa y no_setosa
#usaremos una capa oculta de 6 neuronas (unidades)

#***********inicialización del modelo*******
#inicializamos la matris de pesos [w1] y [w2]
#w1 para los pesos de las entradas
#w2 para los pesos de las salidas de la primera capa oculta

class my_NN(object):
    def __init__(self): #"__init__()" es el constructor en python para inicializar vaores de variables
                        #el "self" hace referencia al nombre del objeto
        self.input=4 # número de columnas (features)
        self.output=1 #  número de clases. En nuestro caso solo 1
        self.hidden_units=6 # capa oculta de 6 neuronas
        
        #inicializamos la matriz de pesos
        np.random.seed(1) #nuestra semilla será 1 (obtendremos  el mismo conjunto de números aleatorios para la misma semilla)
        
        #Peso1: w1. va de la capa de entrada a la primera capa oculta
        self.w1=np.random.randn(self.input,self.hidden_units) #una matriz de 4X6 # el "randn" retorna un array
        #Peso2: w2. va de la primera capa oculta a la capa de salida
        self.w2=np.random.randn(self.hidden_units, self.output) #matriz de 6x1
    
    #Nuestra función de activación será la función sigmoidea
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    #hacemos la multiplicacoón de matrices y los resultados los pasamos por la función de activación
    def _forward_propagation(self,X):
        self.z2=np.dot(self.w1.T,X.T) # el atributo "T" es la transpuesta del array (multiplicamos las entradas con el peso)
        self.a2=self._sigmoid(self.z2) #aplicamos la función sigmoidea a la multiplicación vectorial
        self.z3=np.dot(self.w2.T,self.a2) #multiplicamos los pesos "W2" con la salida "a2" de la primera capa oculta 
        self.a3=self._sigmoid(self.z3) #aplicamos la función sigmoidea a la multiplicación vectorial
        return self.a3
    
    
    #implementamos la función de pérdida
    #usamos la función de perdida que sirve solo para problemas de clasificación binaria: J(0) 
    # se parece al error cuadrático medio
    def _loss(self,predict, y):
        print ("predict: ", predict)
        m=y.shape[0] #"shape[0]" es el numero de de registros (rows)
        logprobs = np.multiply(np.log(predict),y) + np.multiply((1-y),np.log(1-predict))  
        loss=-np.sum(logprobs)/m
        return loss
    
    def _sigmoid_prime(self, x):
        return self._sigmoid(x)*(1-self._sigmoid(x))
    
    #Implementamos la propagación hacia atrás
    def _backward_propagation(self,X, y):
        predict=self._forward_propagation(X)
        m=X.shape[0]
        delta3=predict-y
        
        dz3=np.multiply(delta3,self._sigmoid_prime(self.z3))
        self.dw2= (1/m)*np.sum(np.multiply(self.a2,dz3),axis=1).reshape(self.w2.shape)
        
        delta2=delta3*self.w2*self._sigmoid_prime(self.z2)
        self.dw1=(1/m)*np.dot(X.T,delta2.T)
        
    #actualizamos los pesos w1 y w2 con los resultados de las derivadas parciales de J(0).
    def _update(self, learning_rate=1.2): #mandamos el índice de aprendizaje
        self.w1=self.w1 - learning_rate*self.dw1
        self.w2=self.w2 - learning_rate*self.dw2
        
    #*****Ahora implementamos la etapa de entrenamiento y testeo********************
    
    #con la matriz de pesos W, podemos ejecutar la propagación hacia adelante y obtener el valor final,
    #luego podemos usar ese valor para la predicción
    #si el valor final es >=0.5 predecirá que la clase es "setosa"
    #si el valor final es <0.5 predecirá que la clase  no es setosa
    def train (self, X,y, iteration=33): #haremos 33 iteraciones o "epocas"
        for i in range(iteration):
            print(f"iteración [{i}]")
            #print("X que ingresa: ",X)
            y_hat=self._forward_propagation(X)
            #print("y_hat:",y_hat) 
            loss=self._loss(y_hat, y)
            self._backward_propagation(X, y)
            self._update()
            if i%10==0:
                print(f"loss {i}: {loss}")
    
    
    def predict (self,X):
        y_hat=self._forward_propagation(X)
        y_hat=[1 if i[0]>=0.5 else 0 for i in y_hat.T]
        return np.array(y_hat)
    
    def score(self, predict, y):
        cnt =np.sum(predict==y)
        return (cnt/len(y))*100
    
    
#***********modelo de prueba*********************
from sklearn.model_selection import train_test_split
if __name__=='__main__':
    train_X,test_X,train_y, test_y = train_test_split(X,y_modif ,test_size=0.25) #el 25% será los datos de prueba. #enviamos nuestro "X" e "y" para que haga la división
    clr=my_NN() #inicializamos el modelo
    print("inicio del entrenamiento....")
    clr.train(train_X, train_y) #entrenamos el modelo
    print ("fin del entrenamiento")
    pre_y=clr.predict(test_X) #predecimos con los datos de prueba
    score=clr.score(pre_y,test_y) #obtenemos el indice de exactitud (accuracy score)
    print ('Datos de la predicción: ', pre_y)
    print ('Datos reales', test_y)
    print ("indice de exactitud (accuracy score): ", score)
   
'''
Datos de la predicción:  [0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0 1 0 1 1 0 0 0 1]
Datos reales             [0 1 1 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 0 0 1 0 1 1 0 1 1 1]
indice de exactitud (accuracy score):  86.8421052631579
'''

'''
np.random.seed(0) 
w1=np.random.randn(4,6) #una matriz de 4X6 # el "randn" retorna un array
w2=np.random.randn(6, 1) #matriz de 6x1
print("w1:",w1)
print("w2:",w2)
'''

