# # Variables to decide the information that will be included in the final dataset
#
# * per: Pertentage of the data that will be used.
# * folder: name of the separated datasets
# * folder_per: Percentage of data that will be send to each dataset.
# * batch size: Number of events in each .npz file
per = 100
batch_size = 1000
folders=["train","validation","test"]
folder_per=[70,20,10]
check_per=True 
clean=False

import re
import math
import time
import glob
import numpy as np
import pandas as pd
import os
import random
import multiprocessing

# Warning: Be careful when editing shutil instruction. They can errase everything in disk if not handled correctly
import shutil

# For multicore
import multiprocessing
from multiprocessing import Process
cores = (multiprocessing.cpu_count()-4)

# Create directory for different datasets
try:
    os.mkdir('data/'+str(per)+'_data/')
except FileExistsError:
    print('Directory of datasets already exists')

# Count the total events
total_events = 0
names = glob.glob('data/data_fin/*.npz')
# Random shuffling of the names
random.shuffle(names)
for name in names:
    # Loop throught all the files
    npzfile = np.load(name)
    y = npzfile['y']
    total_events+=y.shape[0]


# Display total
print("There are "+str(total_events)+" in total.")
print("That will be separated into:")
for a in range(len(folders)):
    print(str(folders[a])+' dataset with '+str(folder_per[a])+'%')
    try:
        os.mkdir('data/'+str(per)+'_data/'+str(folders[a])+'_data')
    except FileExistsError:
        print("Folder already exists")
    print("")

    

# Go thought all the files
for name in names:
    npzfile=np.load(name)
    filename=name[-15:]
    X1=npzfile['X1']
    X2=npzfile['X2']
    Y=npzfile['y']
    # Create an index with
    index = np.arange(len(Y))
    np.random.shuffle(index) 
    last=0
    for a in range(len(folders)):
        # Create an index for each one of the sets
        top=int(np.ceil(folder_per[a]*len(Y)/100))
        data_index=index[last:(top+last)]
        last = last + top
        # Now save the X1s and Ys using te data_index in a file in the corresponding folder
        np.savez('data/'+str(per)+'_data/'+str(folders[a])+'_data/'+str(filename),X1=X1[data_index],X2=X2[data_index],y=Y[data_index])



# Delete repeated information
if clean:
    shutil.rmtree('data/data_fin')


# Check the separated data by counting the events
if check_per:
    n_events=[0,0,0]
    for a in range(len(folders)):
        names_data=glob.glob('data/'+str(per)+'_data/'+str(folders[a])+'_data/*.npz')
        for name in names_data:
            npzfile = np.load(name)
            y = npzfile['y']
            n_events[a]+=y.shape[0]
        print("There are "+str(n_events[a])+" events in "+str(folders[a])+" folder, which is "+str((n_events[a]/total_events)*100)+"% of the total events") 
        print("The percentage of that folder should be "+str(folder_per[a])+"%\n")


# function that will make files with a fixed size
def separate(folder,number,per,size):
    # Get event names
    file_dir="data/"+str(per)+"_data/"+str(folders[a])+"_data/*.npz"
    names = glob.glob(file_dir)
    
    #cargo el primer fichero para inicializar las variables
    npzfile = np.load(names[0])
    x1, x2, y = npzfile['X1'], npzfile['X2'], npzfile['y']
    aux_len = x1.shape[0] #longitud auxiliar
    tot_len = x1.shape[0] #longitud total (numero total de eventos que selecciono)
    j=0 #contador de vueltas
    i=0
    total = int(number*per/100)


    for name in names[1:]:
        i+=1
        npzfile = np.load(name)
        aux_x1, aux_x2, aux_y = npzfile['X1'], npzfile['X2'], npzfile['y']
        aux_len += aux_x1.shape[0]
        tot_len += aux_x1.shape[0]

        if ((aux_len <= size) and (tot_len < total)):
            x1, x2, y = np.append(x1,aux_x1,axis=0), np.append(x2,aux_x2,axis=0), np.append(y,aux_y,axis=0)
        elif ((aux_len > size) and (tot_len < total)):
            j+=1
            diff = size-x1.shape[0]
            x1, x2, y = np.append(x1,aux_x1[:diff],axis=0), np.append(x2,aux_x2[:diff],axis=0), np.append(y,aux_y[:diff],axis=0)
            np.savez('data/'+str(per)+'_data/'+str(folder)+'_data_fixed/'+str(folder)+'_dataset_%s.npz' %(str(j)),X1=x1, X2=x2, y=y)
            x1, x2, y = aux_x1[diff:], aux_x2[diff:], aux_y[diff:]
            aux_len=x1.shape[0]
        elif (tot_len>=total):
            j+=1
            diff = total-(tot_len-aux_x1.shape[0])
            x1, x2, y = np.append(x1,aux_x1[:diff],axis=0), np.append(x2,aux_x2[:diff],axis=0), np.append(y,aux_y[:diff],axis=0)
            np.savez('data/'+str(per)+'_data/'+str(folder)+'_data_fixed/'+str(folder)+'_dataset_%s.npz' %(str(j)),X1=x1, X2=x2, y=y)
            break

# Create folders with fixed size data
for a in range(len(folders)):
    try:
        os.mkdir('data/'+str(per)+'_data/'+str(folders[a])+'_data_fixed')
    except FileExistsError:
        print('')

# # Separate data is not working in parallel right now
# # Run a separate function per dataset
# # Open a new pool
# pool = multiprocessing.Pool(processes=cores) 
# # Spawn one process per dataset
# # NOTE: Run in a for loop in the future, may be usefull to run eval
# p1 = Process(target=separate,args=(folders[0],n_events[0],per,batch_size))
# p1.start()

# p2 = Process(target=separate,args=(folders[1],n_events[1],per,batch_size))
# p2.start()

# p3 = Process(target=separate,args=(folders[2],n_events[2],per,batch_size))
# p3.start()

# p1.join()    
# p2.join()  
# p3.join() 

for a in range(len(folders)):
    separate(folders[a],n_events[a],per,batch_size)

# Check the separated data by counting the events
if check_per:
    n_events=[0,0,0]
    for a in range(len(folders)):
        names_data=glob.glob('data/'+str(per)+'_data/'+str(folders[a])+'_data_fixed/*.npz')
        for name in names_data:
            npzfile = np.load(name)
            y = npzfile['y']
            n_events[a]+=y.shape[0]
        print("There are "+str(n_events[a])+" events in "+str(folders[a])+" folder, which is "+str((n_events[a]/total_events)*100)+"% of the total events") 
        print("The percentage of that folder should be "+str(folder_per[a])+"%\n")


# Delete repeated data
if clean:
    for a in range(len(folders)):
        shutil.rmtree('data/'+str(per)+'_data/'+str(folders[a])+'_data_fixed')
        
# Normalize data, only need to normalize the X1 tensor
# Get mean and standart deviation from the train data
names = glob.glob("data/"+str(per)+"_data/train_data_fixed/*.npz")

count = 0
total = n_events[0]
print('There are '+str(total)+' events in the train dataset')    
print('In '+str(len(names))+' files')
print("Normalizing data...")
count = 0

npzfile = np.load(names[0])
Xshape = npzfile["X1"].shape[3]
count += 1

#variable para hacer la media a nivel de set de X1.
#necesito (suma de RGB, suma**2 de RGB, suma de gris, suma**2 de gris)
# X1 puede tener 4 componentes o dos
suma = np.zeros(2, dtype=np.float32)
suma[0]=np.sum(npzfile["X1"][:,:,:,:])
suma[1]=np.sum((npzfile["X1"][:,:,:,:])**2)

for name in names[1:]:

    npzfile = np.load(name)
    
    #suma de RGB, etc
    suma_aux = np.zeros(2, dtype=np.float32)
    suma_aux[0]=np.sum(npzfile["X1"][:,:,:,:Xshape])
    suma_aux[1]=np.sum((npzfile["X1"][:,:,:,:Xshape])**2)

    suma += suma_aux
    count += 1

#calculo a mano la media y la desviacion estandar para X1
mean_x1_rgb = suma[0]/(total*npzfile["X1"].shape[1]*npzfile["X1"].shape[2]*(Xshape))
std_x1_rgb = (suma[1]/(total*npzfile["X1"].shape[1]*npzfile["X1"].shape[2]*(Xshape))-mean_x1_rgb**2)**(0.5)

mean =np.array([mean_x1_rgb])
std = np.array([std_x1_rgb])

np.savez("mean_std.npz", m = mean, s = std)
print("The mean and std obtained:")
print(mean, std)
print("\n")

def z_score_norm (data, mean, std):
    assert (data.ndim) >= 1 , "data should be a vector"
    return ((data - mean) / std)

# # Now apply the mean and std to all the datasets
# Data norm is where the normalized data will be stored
try:
    os.mkdir('data_norm')
except:
    print('El directorio "data_norm" ya existe')

# Create separate directories
for a in range(len(folders)):
    try:
        os.mkdir('data_norm/'+str(folders[a])+'_data_fixed')
    except FileExistsError:
        print('The directory "data_norm/'+str(folders[a])+'_data_'+ str(per)+'" already exists')

def par_norm(name):
    
    # Get the mean and std
    npzfile = np.load("mean_std.npz")
    mean = npzfile["m"]
    std = npzfile["s"]
    
    # Read the correpondig file
    #print("Reading: %s" % (name))
    npzfile = np.load(name)
    
    # # Won't normalize values from X2 or Y vectors.
    x_2 = npzfile["X2"]
    # x_2[:,0] = z_score_norm(x_2[:,0], mean[0], std[0])
    # x_2[:,1] = z_score_norm(x_2[:,1], mean[1], std[1])
    # x_2[:,2] = z_score_norm(x_2[:,2], mean[2], std[2])
    # x_2[:,3] = z_score_norm(x_2[:,3], mean[3], std[3])
    
    y = npzfile["y"]
    # y[:,0] = z_score_norm(y[:,0], mean[4], std[4])
    # y[:,1] = z_score_norm(y[:,1], mean[5], std[5])
    # y[:,2] = z_score_norm(y[:,2], mean[6], std[6])
    # y[:,3] = z_score_norm(y[:,3], mean[7], std[7])
    # y[:,4] = z_score_norm(y[:,4], mean[8], std[8])
    
    x_1 = npzfile["X1"]
    Xshape = npzfile["X1"].shape[3]
    x_1[:,:,:,:Xshape] = z_score_norm(x_1[:,:,:,:(Xshape)], mean,std)
    np.savez('data_norm/'+name[14:], X1=x_1, X2=x_2, y=y)

#from scipy import stats

# Add to names all the files that will be normalized
names = glob.glob("data/"+str(per)+"_data/train_data_fixed/*.npz")
names += glob.glob("data/"+str(per)+"_data/test_data_fixed/*.npz")
names += glob.glob("data/"+str(per)+"_data/validation_data_fixed/*.npz")
print('Normalizing '+str(len(names))+' files')


#Open a new pool
pool = multiprocessing.Pool(processes=cores) 
# Loop the parallel function thought all the files
pool.map(par_norm, names) 
pool.close() 
pool.join()    
print('Finished normalizing all data\n')

if check_per:
    # Get mean and standart deviation from the train data
    names = glob.glob("data_norm/train_data_fixed/*.npz")

    count = 0
    # total = n_events[0]
    # print('There are '+str(total)+' events in the train dataset')    
    # print('In '+str(len(names))+' files')
    #full_data = np.zeros((total, 9), dtype = np.float32)
    count = 0

    print("Reading: %s" % (names[0]))
    npzfile = np.load(names[0])
    Xshape = npzfile["X1"].shape[3]
    print(Xshape)
    count += 1

    suma = np.zeros(2, dtype=np.float32)
    suma[0]=np.sum(npzfile["X1"][:,:,:,:(Xshape)])
    suma[1]=np.sum((npzfile["X1"][:,:,:,:(Xshape)])**2)

    for name in names[1:]:

        npzfile = np.load(name)

        #suma de RGB, etc
        suma_aux = np.zeros(2, dtype=np.float32)
        suma_aux[0]=np.sum(npzfile["X1"][:,:,:,:(Xshape)])
        suma_aux[1]=np.sum((npzfile["X1"][:,:,:,:(Xshape)])**2)

        suma += suma_aux
        count += 1

    #calculo a mano la media y la desviacion estandar para X1
    mean_x1_rgb = suma[0]/(total*npzfile["X1"].shape[1]*npzfile["X1"].shape[2]*(Xshape))
    std_x1_rgb = (suma[1]/(total*npzfile["X1"].shape[1]*npzfile["X1"].shape[2]*(Xshape))-mean_x1_rgb**2)**(0.5)

    mean =np.array([mean_x1_rgb])
    std = np.array([std_x1_rgb])
    print("\n")
    print("The mean and std after normalizing the training data is: ")
    print(mean, std)
    print("Ideally, mean should be zero and std 1")

if clean:
    # Deleting data directory
    shutil.rmtree('data')
    # Deleting events directory
    shutil.rmtree('events')
