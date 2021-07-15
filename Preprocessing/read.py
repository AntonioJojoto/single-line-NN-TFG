import re
import math
import time
import glob
import gzip
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import multiprocessing
#some global variables
up_tol = 600 #900.0 changed to have lower cost
low_tol = -200 #-300.0
Ts = 5 #3 ns
Nt = int((up_tol+low_tol)/ Ts)

def muon_data(X, Y, Z, X1, Y1, Z1, energy, type):
#(X, Y, Z) -> dirección
#(X1,Y1,Z1) -> posicion de generacion
    return(pd.DataFrame(data = [float(X), float(Y), float(Z), float(X1), float(Y1), float(Z1),energy, type],
                        index = ["X","Y","Z","X1","Y1","Z1","Energy","Type"],
                        columns = ["Muon"]))

def neutrino_data(X, Y, Z, energy, type):
    return(pd.DataFrame(data = [float(X), float(Y), float(Z), energy, type],
                        index = ["X","Y","Z","Energy","Type"],
                        columns = ["Neutrino"]))

def aafit_data(X, Y, Z, lamb, beta):
    #lamb = lambda -> calidad de reconstrucción, beta-> precisión en la dirección
    return(pd.DataFrame(data = [float(X), float(Y), float(Z), float(lamb), float(beta)],
                        index = ["X","Y","Z","Lambda","Beta"],
                        columns = ["aafit"]))

def bbfit_track_data(X, Y, Z, zc, dc, f_quality):
#zc -> componente z del punto mas cercano a la linea detectora, dc -> distancia a la linea
    return(pd.DataFrame(data = [None,None, float(Z), zc, dc, f_quality],
                        index = ["X","Y","Z","zc","dc","Reconstruction quality"],
                        columns = ["bbfit track"]))

def selected_hit_data(floor, x, y, z, _x, _y, _z, t, A ):
    #(x,y,z) -> posicion, (_x,_y,_z) -> orientacion , t -> tiempo, A-> amplitud
    return(pd.DataFrame(data = [[int(floor),float(x),float(y),float(z),float(_x),float(_y),float(_z),float(t),float(A)]],
                       columns = ["floor","x","y","z","ux","uy","uz","t","A"]))

def event_data(run, numu, event_id, weights, nu, muon, bbfit_track, aafit, hits, bad_hits, Tr, linea):
    return(pd.DataFrame(data = [[run, numu, int(event_id), weights, nu, muon, bbfit_track, aafit, hits, bad_hits, Tr, linea]],
                       columns = ["run","numu","Event ID","Weights data","Nu data","Muon data","bbfit track data","aafit data","Selected hits","Selected bad hits","Tr", "linea"]))

#this function uses the selected hit data to actually get the selected hits DataFrame
def select_hits(hits, s_hits):
    result = pd.DataFrame()
    Tr = float(s_hits[0][11])
    for i in range(1,len(s_hits)):
        if float(s_hits[i][11]) < Tr :
            Tr = float(s_hits[i][11])
    for i in range(0,len(hits)):
            if s_hits[0][2] == hits[i][2]: # just if they are in the same line 
                if ((float(hits[i][11])) > (Tr + low_tol) and (float(hits[i][11])) <= (Tr + up_tol)): #then we take those hits E (tr - 500, tr + 1500)
                    result = result.append(selected_hit_data(hits[i][3], hits[i][5], hits[i][6], hits[i][7], hits[i][8], hits[i][9], hits[i][10], hits[i][11], hits[i][12]), ignore_index = True)
                    #print(hits[i][1])
    return result,Tr

def select_bad_hits(hits, s_hits):
    result = pd.DataFrame()
    # Cojo el tiempo de referencia
    Tr = float(s_hits[0][11])
    for i in range(1,len(s_hits)):
        if float(s_hits[i][11]) < Tr :
            Tr = float(s_hits[i][11])
    # Selecciono la linea 'mala'
    good = int(s_hits[0][2])
    a = list(range(1,13))
    a.remove(good)
    bad = a[random.randrange(11)]
    # Itero sobre los hits para obtener la seleccion final
    for i in range(0,len(hits)):
            if bad == int(hits[i][2]): # just if they are in the chosen at random 'bad' line 
                if ((float(hits[i][11])) > (Tr + low_tol) and (float(hits[i][11])) <= (Tr + up_tol)): #then we take those hits E (tr - 500, tr + 1500)
                    result = result.append(selected_hit_data(hits[i][3], hits[i][5], hits[i][6], hits[i][7], hits[i][8], hits[i][9], hits[i][10], hits[i][11], hits[i][12]), ignore_index = True)
                    #print(hits[i][1])
    return result,Tr


def read_event (run, numu, fhand):
    while True :
            #reading blank lines until we can get the id of the event
        linea = fhand.readline()
        if re.search("^start_event", linea) != None:
            id = re.findall('\s*(\S+)\s*',linea)
            id = id[1]
            break
        if len(linea) == 0:
            return pd.DataFrame(data = [1],columns = ["out"])
    #now we skip the lines until we find the nu line
    while True :
        linea = fhand.readline()
        if re.search('^weights', linea) != None :
            w = re.findall('\s*(\S+)\s*',linea)
            w = pd.DataFrame(data = w[1:],columns = ["Weigths"])
            linea = fhand.readline()
            break
    #if there is some missing data, the line of data will be equal to None
    if re.search('^nu', linea) != None :
        _nu = re.findall('\s*(\S+)\s*',linea)
        _nu = neutrino_data(_nu[1], _nu[2], _nu[3], _nu[7], _nu[8])
        linea = fhand.readline()
    if re.search('^muon', linea) != None :
        _muon = re.findall('\s*(\S+)\s*',linea) # we take data from the reade line
        _muon = muon_data(_muon[1], _muon[2], _muon[3], _muon[4], _muon[5], _muon[6], _muon[7], _muon[8]) #we init the muon class with selected data
        linea = fhand.readline()
    else:
        _muon = None
    if re.search('^aafit', linea) != None :
        _aafit = re.findall('\s*(\S+)\s*',linea)
        _aafit = aafit_data(_aafit[1], _aafit[2], _aafit[3], _aafit[7], _aafit[8])
        linea = fhand.readline()
    else:
        _aafit = None
    if re.search('^bbfit_track', linea) != None :
        _bbfit_track = re.findall('\s*(\S+)\s*',linea) #es el unico lugar que tiene valores nan
        if _bbfit_track[1] !=  "nan" or _bbfit_track[2] != "nan":
            return pd.DataFrame()
        _bbfit_track = bbfit_track_data(_bbfit_track[1], _bbfit_track[2], _bbfit_track[3], _bbfit_track[4], _bbfit_track[6], _bbfit_track[7])
    else:
        _bbfit_track = None
    n_hits = 0
    while True :
        linea = fhand.readline()
        if re.search('^hit', linea) != None :
            hits = list()
            hits.insert(n_hits, re.findall('\s*(\S+)\s*',linea))
            break
    while True :
        linea = fhand.readline()
        if re.search('.*selected pulses:', linea) != None :
            break
        else:
            n_hits = n_hits + 1
            hits.insert(n_hits, re.findall('\s*(\S+)\s*',linea))
    s_hits = list()
    n_s_hits = 0 #number of selected hits
    while True :
        linea = fhand.readline()
        if re.search('.*end_event', linea) != None :
            #print("fin del evento")
            break
        else:
            s_hits.insert(n_hits, re.findall('\s*(\S+)\s*',linea))
            n_s_hits = n_s_hits + 1
    selected,Tr = select_hits(hits, s_hits)
    selected_bad,Tr2 = select_bad_hits(hits, s_hits)
    return (event_data(run, numu, id, w, _nu, _muon, _bbfit_track, _aafit, selected, selected_bad, Tr, int(s_hits[0][2])))
