# # Variables to decide the behaviour of the program
# 
# * preread: True if the events have been read.
# * classified: True if the events have been separated into the three separate datasets
# * processed: True if the kernel has been applied to the program
# * RIGEL: Run with all the threats, if not, it will leave one core free.

preread = False
processed = False
clean= False
RIGEL = False
BadLine = False

# # Variables to decide the information that will be included in the final dataset
#
# * RGB: True if the convolution will use de module of the RBG
# * X2: True if the files should contain the X2 tensor
# * Y3: True if the files should have the XYZ components

RGB_Mod = True
X2_comp = True
Y3 = False


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

# Warning: Be careful when editing shutil instruction. They can errase everything in disk if not handled correctly
import shutil
# Locally installed from https://github.com/jmetzen/kernel_regression
from kernel_regression import KernelRegression

# Import functions from predefined libraries
from read import *
from preprocessing import *
# For multicore
import multiprocessing
from multiprocessing import Process
cores = 10
# if not(RIGEL):
    # cores-=2

#some global variables
up_tol = 600 #900.0 changed to have lower cost
low_tol = -200 #-300.0
Ts = 5 #3 ns
Nt = int((up_tol+low_tol)/ Ts)



#creo el directorio para los datos procesados
print("Creating the events directory")
try:
    os.mkdir('events')
except FileExistsError:
    print('Folder events already exists')

# Used to save time when reading all the .gz and saving to .plk in events
def par_read(name):
    events_aux = pd.DataFrame() # DataFrame auxiliar para cada archivo de eventos
    run = re.findall("_([0-9]+)_", name)[0]
    if re.search("[0-9]_(numu)_", name) != None:
        numu = 1
    else:
        numu = 0
    with gzip.open(name, 'rt') as fhand:
        while True:
            aux = read_event(run, numu, fhand)
            if aux.empty:
                continue
            elif aux.columns[0] == "out":
                break
            else:
                events_aux = events_aux.append(aux, ignore_index = True)
    # Save the new file, simplifying the name
    if not (events_aux.empty):
        if numu:
            events_aux.to_pickle("events/%s.pkl" %(name[17:28]))
        else:
            events_aux.to_pickle("events/%s.pkl" %(name[17:29]))

# Read data using more cores
if not preread:
    names = glob.glob("i2ascii-files/*.gz") # coge todos los archivos que terminen en '.gz'
    # Open a new pool
    pool = multiprocessing.Pool(processes=cores) 
    # Loop the par_read function thought all the files in the i2ascii-files/ directory
    pool.map(par_read, names) 
    pool.close() 
    pool.join()    
    print('Finished loading data')
        

# # Data Preprocessing
# Directory data is where all the processed data will be stored
try:
    os.mkdir('data')
except FileExistsError:
    print('"data" directory already exists')
try:
    os.mkdir('data/raw')
except FileExistsError:
    print('"data" directory already exists')
# ## Functions to run preprocessing parallel

# Called when doing parallel preprocessing, opens each .plk file and applies preiously defined function to
# do the offline convolution
def par_preprocess(name):
    events = pd.read_pickle(name)
    
    #reseteo el indice, porque se guardaron sin orden,
    #si estan ordenados, esta linea puede dejarse sin problemas, no haria nada
    events = events.reset_index(drop=True)
    
    tens, y = regression(events)
    greys = np.zeros((tens.shape[0], tens.shape[1], tens.shape[2]), dtype = np.float32) 
    for i in range(greys.shape[0]):
        # If RGB module is derised, then greys will be the module only
        if RGB_Mod:
            greys[i] = np.sqrt(np.sum(tens[i]**2,axis=2))
        else:
            greys[i] = rgb_merge(tens[i])
    greys = greys.reshape(greys.shape[0], greys.shape[1], greys.shape[2], 1)
    
    kernel = np.ones((3,20,1,1))

    # Offline conv, not necesary if online conv will be used
    # conv_out_tf = conv2d(greys, kernel)
    # conv_out = conv_out_tf.numpy()
    # conv_out = conv_out.reshape(greys.shape[0], greys.shape[1], greys.shape[2])
    
    if RGB_Mod:
        X = np.zeros((tens.shape[0], tens.shape[1], tens.shape[2], 1), dtype = np.float32)
        X[:,:,:,:] = greys
        # X[:,:,:,1] = conv_out
    else:
        X = np.zeros((tens.shape[0], tens.shape[1], tens.shape[2], 3), dtype = np.float32)
        X[:,:,:,:] = tens
        # X[:,:,:,3] = conv_out
        

    # Finally, set the output according to the user input.
    # True gives the Y matrix the X,Y,Z and energy components of the neutrino
    # False only gives the Z and energy components

    if Y3:
        # Only deletes the X*Y row
        y=np.delete(y,2,1)
    else:
        # Delete the X,Y and X*Y rows
        y=np.delete(y,[0,1,2],1)

    # Handling the file name
    filename="data/raw/"+name[7:-4]+".npz"
    np.savez(filename, X=X, y=y)

# ## Run parallel preprocessing
if not processed:
    # Get event names
    names = glob.glob("events/*.pkl")
    names = sorted(names)
    # Open a new pool
    pool = multiprocessing.Pool(processes=cores) 
    # Loop the par_read function thought all the files in the i2ascii-files/ directory
    pool.map(par_preprocess, names) 
    pool.close() 
    pool.join()    


# # Adding X2 (vector input 2)
# Directories data is where X2 data will be stored
try:
    os.mkdir('data/data_fin')
except FileExistsError:
    print('Directory "data_fin" already exists')

#Limites de las lineas
z_min = -171.56
z_max = 177.16
# ## Adding get_X2 parallel function
def get_x2_par(name):
    df = pd.read_pickle(name)
    df = df.reset_index(drop=True)
    
    x_2 = np.zeros((df.shape[0], 7), dtype = np.float32)
    
    for ind in range(df.shape[0]):
        
        qx, qy, qz = df.at[df.index[ind],"Muon data"].at["X1","Muon"], df.at[df.index[ind],"Muon data"].at["Y1","Muon"], df.at[df.index[ind],"Muon data"].at["Z1","Muon"]
        ux, uy, uz = df.at[df.index[ind],"Muon data"].at["X","Muon"], df.at[df.index[ind],"Muon data"].at["Y","Muon"], df.at[df.index[ind],"Muon data"].at["Z","Muon"]
        
        if BadLine:
            hits = df.at[df.index[ind], "Selected bad hits"]
        else:
            hits = df.at[df.index[ind], "Selected hits"]
        
        # Calculo de Z ponderada con la amplitud de los hits
        pond = (hits['A']*hits['z']).sum()/hits.shape[0]
        
        # Calculo la posicion XY de las lineas
        equis = 0.0
        why = 0.0
        for ind_hit in range(hits.shape[0]):
            equis += hits.at[ind_hit, "x"]
            why += hits.at[ind_hit,"y"]
        Lx = equis/hits.shape[0]
        Ly = why/hits.shape[0]
        
        # Calculo el Zeta de la linea al que mas se acerca el muon
        zeta_c = qz +uz*((Lx-qx)*ux+(Ly-qy)*uy)/(1-uz**2)
        
        if zeta_c > z_max:
            if uz >= 0.0:
                zeta_c = z_max
            else:
                zeta_c = np.min([qz,z_max])
        elif zeta_c < z_min:
            if uz < 0.0:
                zeta_c = z_min
            else:
                zeta_c = np.max([qz, z_min])
        else:
            if (uz >= 0.0) and (zeta_c < qz):
                zeta_c = np.min([qz, z_max])
            elif (uz < 0.0) and (zeta_c > qz):
                zeta_c = np.max([qz, z_min])
        
        if zeta_c > z_max:
            zeta_c = z_max
        if zeta_c < z_min:
            zeta_c = z_min
        
        # Calculo la distancia teniendo en cuenta que es una semirrecta
        lamb = (Lx-qx)*ux+(Ly-qy)*uy+(zeta_c-qz)*uz
        lamb = np.max([lamb, 0.0])
        dist = np.sqrt((qx+lamb*ux-Lx)**2+(qy+lamb*uy-Ly)**2+(qz+lamb*uz-zeta_c)**2)
        
        # Guardo los valores
        x_2[ind, 1] = dist
        x_2[ind, 2] = zeta_c
        x_2[ind, 3] = pond
        
        x_2[ind, 0] = df.at[df.index[ind],"bbfit track data"].at["Z","bbfit track"]
        x_2[ind, 4] = float(df.at[df.index[ind],"bbfit track data"].at["zc","bbfit track"])
        x_2[ind, 5] = float(df.at[df.index[ind],"bbfit track data"].at["dc","bbfit track"])
        x_2[ind, 6] = float(df.at[df.index[ind],"bbfit track data"].at["Reconstruction quality","bbfit track"])
    filename="data/raw/"+name[7:-4]+".npz"
    npzfile = np.load(filename)
    filename2="data/data_fin/"+name[7:-4]+".npz"
    if X2_comp:
        np.savez(filename2, X1=npzfile["X"], X2=x_2 ,y=npzfile["y"])
    else:
        np.savez(filename2, X1=npzfile["X"], y=npzfile["y"])
   
# ## Get X2 in parallel
# Get event names
names = glob.glob("events/*.pkl") 

# Open a new pool
pool = multiprocessing.Pool(processes=cores) 
# Loop the parallel function thought all the files
pool.map(get_x2_par, names) 
pool.close() 
pool.join()    
print('Finished Getting X2s')

# # Deleting events with negative amplitude A 
# ## Adding delete_negative parallel function

def delete_negative(name):
    
    df = pd.read_pickle(name)
    df = df.reset_index(drop=True)
    
    msk = np.ones((df.shape[0]), dtype = bool)
    
    for ind in range(df.shape[0]):
        if df.at[df.index[ind],"Selected hits"].sort_values(by=['A'], ignore_index = True).at[0,"A"] < 0:
            msk[ind] = 0
    
    filename="data/data_fin/"+name[7:-4]+".npz"
    npzfile = np.load(filename)

        
    if X2_comp:
        np.savez(filename, X1=npzfile["X1"][msk], X2=npzfile["X2"][msk], y=npzfile["y"][msk])
    else:
        np.savez(filename, X1=npzfile["X1"][msk], y=npzfile["y"][msk])

# ## Delete negative events in parallel

# Get event names
names = glob.glob("events/*.pkl")

# Open a new pool
pool = multiprocessing.Pool(processes=cores) 
# Loop the parallel function thought all the files
pool.map(delete_negative, names) 
pool.close() 
pool.join()    
print('Finished deleting negative amplitude events')

# ### Deleting processed data, back up if you want
if clean:
    shutil.rmtree("data/raw/")
    shutil.rmtree("events")
