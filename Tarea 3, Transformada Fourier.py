# -*- coding: utf-8 -*-
"""
Spyder Editor

Física Computacional

Valentina Campos Aguilar
Luis ALfredo Guerrero Camacho

TRANSFORMADA DE FOURIER
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Se definen los parámetros de muestreo.
tasa_muestreo = 2048   #número de muestras por unidad de tiempo
deltaT = 1   #unidad de tiempo

# Se define el tamaño del arreglo de muestras. Cantidad de pares de datos. 
nPuntos = deltaT*tasa_muestreo

'''
 Se crea el arreglo de los puntos t, o sea el arreglo del dominio del tiempo a analizar. 
 
 Se usa la función linspace que crea este arreglo estableciendo los puntos t partiendo 
 de 0 hasta la cantidad total de muestras (nPuntos) cada unidad de tiempo (deltaT).  
'''
arreglo_tiempo = np.linspace(0, deltaT, nPuntos)

''' Se parte de tres frecuencias bases para generar 3 señales, los valores de frecuencia
 y magnitud se define aleatoriamente. '''
 
frecuencia1 = 60
magnitud1 = 20

frecuencia2 = 100
magnitud2 = 40

frecuencia3 = 120
magnitud3 = 30

# Se definen las funciones para caracterizar las tres señales. 
señal1 = magnitud1*np.sin(2*np.pi*frecuencia1*arreglo_tiempo)
señal2 = magnitud2*np.sin(2*np.pi*frecuencia2*arreglo_tiempo)
señal3 = magnitud3*np.sin(2*np.pi*frecuencia3*arreglo_tiempo)

# Se general el ruido para la señal original. 
ruido = np.random.normal(0, 20, nPuntos)

''' La señal original estrará dada por la suma de las tres señales base, según el principio
 de superposición de ondas. '''
 
señal_original = señal1 + señal2 + señal3

# La señal ruidosa se obtendrá añadiendo el ruido generado a la señal original.
señal_ruidosa = señal1 + señal2 + señal3 + ruido

'''Se procede a graficar en una misma figura la señal original producida y la misma con 
el ruido aplicado.  '''

#Formato figura en general.
fig, (ax1, ax2) = plt.subplots(1, 2, dpi = 120, sharey=True)
fig.suptitle('Gráfico 1. Comparación entre señal original y señal ruidosa', fontsize=12)
#Gráfica señal original.
ax1.plot(arreglo_tiempo[1:100], señal_original[1:100])
ax1.set_title('Señal original')
ax1.set_xlabel('Tiempo [s]')
ax1.set_ylabel('Amplitud')
#Gráfica señal ruidosa
ax2.plot(arreglo_tiempo[1:100], señal_ruidosa[1:100])
ax2.set_title('Señal ruidosa')
ax2.set_xlabel('Tiempo [s]')
plt.show()


# Aplicación de la Transformada de Fourier a la señal ruidosa:
    
''' Se crea el arreglo de las frecuencias a analizar con la función linspace partiendo del 
valor 0 hasta la mitad de la tasa de muestreo registrando un punto de frecuencia cada 
cantidad de pares de datos entre 2 (dominio de frecuencias). ''' 
puntos_frecuencia = np.linspace(0.0, 1024, int(nPuntos/2))

# Se aplica la Transformada Rápida de Fourier a la señal ruidosa. 
señal_transformada = sp.fft.fft(señal_ruidosa)

#Se crea el arreglo de amplitudes normalizando la señal evaluada en la mitad de los puntos.
amplitudes = (2/nPuntos)*np.abs(señal_transformada[0:np.int(nPuntos/2)])


# Se grafica la señal ruidosa en el dominio de la frecuencia. 
fig, ax = plt.subplots(dpi=120)
ax.plot(puntos_frecuencia, amplitudes)
ax.set_title('Gráfico 2. Señal ruidosa en el dominio de la frecuencia')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud')
ax.set_xticks(np.arange(0,1013,100))
plt.show() 
 

# Filtración del ruido en la señal ruidosa en el dominio de la frecuencia:

''' 
Se crea una función con parámetros amplitud (correspondiente al arreglo de amplitudes d
de la señal por filtrar), umbral (parámetro de filtración) y señal (señal a filtrar).

La misma tendrá como salida la señal filtrada y la gráfica de la misma.
'''

def Filtrar_Señal(amplitud, señal, umbral):
    for i in range(len(amplitud)):
        if amplitud[i] < umbral:
            señal[i] = 0.0+0.0j
            amplitud[i] = 0.0
    fig, ax = plt.subplots(dpi = 120)
    ax.plot(puntos_frecuencia, amplitud)
    ax.set_title('Gráfico 3. Señal filtrada en dominio de la frecuencia')
    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('Amplitud')
    ax.set_xticks(np.arange(0,1013,100))
    plt.show()
    return señal

'''Se define la señal filtrada llamando a la función Filtrar_Señal, donde se escogen como
 parámetros un umbral de 5, la señal a filtrar será la señal transformada evaluada de 0 
 a todos los pares de puntos y el arreglo de amplitudes de la señal transformada. '''

señal_filtrada = Filtrar_Señal(amplitudes,señal_transformada[0:np.int(nPuntos)], 5)


# Aplicación de la Transformada Rápida Inversa de Fourier a la señal filtrada:
 
# Se aplica la transformada inversa a la señal filtrada.     
TFInversa = sp.fft.ifft(señal_filtrada)
# Se define la señal resultante como la parte real de la transformación inversa. 
# Esta es la señal filtrada en el dominio temporal. 
señal_resultante = TFInversa.real

#Se procede a graficar la señal resultante. 
fig, ax = plt.subplots(dpi = 120)
ax.plot(arreglo_tiempo[1:100], señal_resultante[1:100])
ax.set_title('Gráfica 4 \n Señal filtrada')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Amplitud')
plt.show()

