# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:06:16 2022

@author: class
"""
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

celsius = np.array([-40,-10,0,8,12,16,22],dtype=float)

fahrenheit = np.array([-40,15,30,42,53,72,100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'    
)

print("Comenznado entrenamiento...")
historial= modelo.fit(celsius,fahrenheit,epochs=1000,verbose=False)
print("Modelo entrenado")

plt.xlabel("Epoca")
plt.ylabel("Magitud de perdida")
plt.plot(historial.history["loss"])

print("Prediccion")
resultado = modelo.predict([100.0])
print("El resultado es: ", str(resultado),"fahrenheit")