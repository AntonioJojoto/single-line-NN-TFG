import re
import math
import time
import glob
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing
from kernel_regression import KernelRegression
#some global variables
up_tol = 600 #900.0 changed to have lower cost
low_tol = -200 #-300.0
Ts = 5 #3 ns
Nt = int((up_tol+low_tol)/ Ts)

# esta función está bien
def hit_to_pixel(ux, uy, A):
    R = 0;
    G = (2 * np.pi) / 3 # 120 grados
    B = (4 * np.pi) / 3 # 240 grados
    zenith = math.atan2(uy, ux)
    if zenith < 0: # si es negativo se suma 2*pi para ponerlo en positivo
        zenith = zenith + 2 * np.pi
    if zenith > B:
        wr = (G - ((2 * np.pi) - zenith)) / G
    else:
        wr = (G - zenith) / G
    wg = (G - abs(G - zenith)) / G
    wb = (G - abs(B - zenith)) / G
    if wr < 0:
        wr = 0
    if wg < 0:
        wg = 0
    if wb < 0:
        wb = 0
    return (np.array([wr * A, wg * A, wb * A]))

# primero se hace una regresion a las imagenes de colores, cuyo tamaño es (25, t.shape, 3)
# porque hay 25 pisos, una linea temporal con 't.shape' tiempos, y 3 colores
def event_regression (hit_df, tr, event_dt, regression_dt, ksigma):
    
    t = np.round(np.arange(low_tol, up_tol + event_dt, event_dt),1) #because we don't take the low_tol value
    t = np.array(t).reshape(-1, 1) # lo transforma en una matriz Nx1
    x = np.zeros((25, t.shape[0], 3)) # creo la imagen llena de ceros
    
    for ind_hit in range(0, hit_df.shape[0]): # 'for' en el DataFrame de los SelectedHits de un evento
        floor = (hit_df.at[ind_hit,"floor"] - 1)
        event_t_index = int( round( (hit_df.at[ind_hit,"t"] - (tr + low_tol)) / event_dt) )# round 1 because our event dt will be 0.1
        #save the RGB values
        x[floor][event_t_index] += hit_to_pixel(hit_df.at[ind_hit,"ux"], hit_df.at[ind_hit,"uy"], hit_df.at[ind_hit,"A"])      
        # aquí se ha rellenado la imagen con los valores RGB
        
    #we start the regression
    sub_step = int(regression_dt/event_dt)
    t_sub = t[::sub_step] # coge tiempos de 'sub_step' en 'sub_step'
    gkr = KernelRegression(kernel = "rbf", gamma = 1/(2*ksigma**2))
    
    r_gkr = np.zeros((x.shape[0],t_sub.shape[0])) #matriz roja para la regresion
    g_gkr = np.zeros((x.shape[0],t_sub.shape[0])) #matriz verde
    b_gkr = np.zeros((x.shape[0],t_sub.shape[0])) #matriz azul
    rgb_gkr = np.zeros((x.shape[0],t_sub.shape[0],3)) #matriz para todos
    
    for ind_floor in range(0, 25): # se hace la regresion por cada piso
        # Se ha modificado ligeramente el codigo
        # para ver como y por que se ha hecho ver 'code/Data_preprocessing/example_kernel_regression.ipynb'
        gkr.fit(t, x[ind_floor,:,0])
        r_gkr[ind_floor] = gkr.predict(t_sub)
        gkr.fit(t, x[ind_floor,:,1])
        g_gkr[ind_floor] = gkr.predict(t_sub)
        gkr.fit(t, x[ind_floor,:,2])
        b_gkr[ind_floor] = gkr.predict(t_sub)
    
    rgb_gkr[:,:,0] = r_gkr #guardo el rojo
    rgb_gkr[:,:,1] = g_gkr #guardo el verde
    rgb_gkr[:,:,2] = b_gkr #guardo el azul
    
    #we do the normalization 
    # AQUI ES DONDE PONE LOS VALORES ENTRE 0-1, ESTO CREO QUE HAY QUE QUITARLO
    #print(np.min(rgb_gkr), np.max(rgb_gkr))
    #rgb_gkr = (rgb_gkr - np.min(rgb_gkr)) / (np.max(rgb_gkr) - np.min(rgb_gkr))#rgb_gkr / np.max(rgb_gkr)
    
    #print(rgb_gkr.shape, np.min(rgb_gkr), np.max(rgb_gkr))
    return (rgb_gkr)   

def rgb_merge(rgb):
    # Returns the mean of the RGB
    return np.dot(rgb[..., :3], 1/3*np.ones(3))
    #return np.sum(rgb,axis = 2)

def rgb_mod(rgb):
    # Returns the module of the RGB
    return np.sqrt(np.sum(rgb**2,axis=2))

def show_events_grey (greys, event_df):
    plt.close('all')
    # Plot in different subplots
    for i in range(0, event_df.shape[0]):
        plt.figure(figsize=(15, 13))
        X = event_df.at[i, "Nu data"].at["X","Neutrino"]
        Y = event_df.at[i, "Nu data"].at["Y","Neutrino"]
        Z = event_df.at[i, "Nu data"].at["Z","Neutrino"]
        e = event_df.at[i, "Nu data"].at["Energy","Neutrino"]
        t = event_df.at[i, "Nu data"].at["Type","Neutrino"]
        plt.suptitle('Neutrino data: X = %s, Y= %s, , Z= %s, Energy= %s, Type= %s' % (X, Y, Z, e, t), y=0.93)
        plt.tight_layout()
        xtick_step = 200
        plt.imshow(greys[i], vmin=0, vmax=1, cmap= "gray", interpolation = "none", extent=[low_tol - Ts/2, up_tol + Ts/2, greys.shape[1] + 0.5, 0.5], aspect = Ts*np.size(greys, axis = 2)/np.size(greys, axis = 1))
        plt.gca().set_xticks(np.arange(low_tol, up_tol + xtick_step, xtick_step))
        plt.gca().invert_yaxis() 
        plt.title('Grey_scale')
        #plt.savefig("event_%s_image_regressiondt_%s" % (event_df.at[i, "Event ID"],Ts),bbox_inches="tight")
        plt.show()

def conv2d (inp, kernel, stride = 1, padding = "SAME"):
    assert inp.ndim == 4 or kernel.ndim == 4 , "The input arrays must be 4-D"
    inp2 = tf.constant(inp, tf.float32)
    kernel2 = tf.constant(kernel, tf.float32)
    return (tf.nn.conv2d(inp2, kernel2, [1, stride, stride, 1], padding=padding))
def regression (events):
    
    Y = np.zeros((events.shape[0], 5), dtype = np.float32)
    #lstime = time.time()
    #ctime = 0
    
    #print("Starting regression") #empieza la regresion para todo un DF de eventos
    aux = event_regression(events.at[events.index[0], "Selected hits"].sort_values(by=['t'], ignore_index = True), 
                  events.at[events.index[0], "Tr"], 1, Ts, 3) #hace la regresion de un evento
    
    tens = np.zeros((events.shape[0], aux.shape[0], aux.shape[1], aux.shape[2]), dtype = np.float32) 
    tens[0] = aux
    Y[0, 0] = events.at[events.index[0],"Nu data"].at["X","Neutrino"]
    Y[0, 1] = events.at[events.index[0],"Nu data"].at["Y","Neutrino"]
    Y[0, 2] = events.at[events.index[0],"Nu data"].at["X","Neutrino"] * events.at[events.index[0],"Nu data"].at["Y","Neutrino"]
    Y[0, 3] = events.at[events.index[0],"Nu data"].at["Z","Neutrino"]
    Y[0, 4] = events.at[events.index[0],"Nu data"].at["Energy","Neutrino"]
    
    count = 1
    #ctime = (time.time() - lstime)
    #print("Event regression in: %.3f seconds" % (time.time() - lstime))
    #print("%.2f %% completed in: %.3f seconds " % (100*count/events.shape[0], ctime))
    
    #hace la regresion del primero y luego sigue con el resto en un 'for'
    for ind in events.index[1:]:
        
        #lstime = time.time()
        
        #print(ind)
        tens[count] = event_regression(events.at[ind, "Selected hits"].sort_values(by=['t'], ignore_index = True), 
                                    events.at[ind, "Tr"], 1, Ts, 5)
        
        Y[count, 0] = events.at[ind,"Nu data"].at["X","Neutrino"]
        Y[count, 1] = events.at[ind,"Nu data"].at["Y","Neutrino"]
        Y[count, 2] = events.at[ind,"Nu data"].at["X","Neutrino"] * events.at[ind,"Nu data"].at["Y","Neutrino"]
        Y[count, 3] = events.at[ind,"Nu data"].at["Z","Neutrino"]
        Y[count, 4] = events.at[ind,"Nu data"].at["Energy","Neutrino"]
        
        count += 1
        #ctime += (time.time() - lstime)
        #print("Event regression in: %.3f seconds" % (time.time() - lstime))
        #print("%.2f %% completed in: %.3f seconds " % (100*count/events.shape[0], ctime))#events.shape[0], ctime))
    
    return tens, Y #tens= tensor (nº de eventos, pisos, tiempo, rgb); Y= informacion de los eventos
