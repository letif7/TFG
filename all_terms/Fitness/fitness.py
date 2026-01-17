import numpy as np
import re
from ..descproteins import GAAC,EGAAC,CKSAAGP,GDPC,GTPC,BLOSUM62,CTDC,CTDT,CTDD
from Bio.SVDSuperimposer import SVDSuperimposer
from ..EDA_tools.EDAtools import similitud_MC,entropia_descrip_val,entropia_descrip_prob, rmsd_MC, calcular_divergencias
from transformers import EsmTokenizer, EsmModel
import torch
import pyrosetta
from pyrosetta import rosetta

#Esta funcion es para crear el tipo Pose, que neceista Rosetta para calcular la energía: 
def _coords_to_pose(self, sequence, coords_3d):
    pose = pyrosetta.Pose()
    pyrosetta.pose_from_sequence(pose, sequence)

    atom_idx = 0
    for i in range(1, pose.total_residue() + 1):
        for atom in ["N", "CA", "C"]:
            x, y, z = coords_3d[atom_idx]
            pose.residue(i).set_xyz(
                atom,
                pyrosetta.rosetta.numeric.xyzVector_double_t(x, y, z)
            )
            atom_idx += 1

    return pose

#y este es la funcion, que dado el Pose , calcula la energía
def _calculate_design_energy(self, sequence, coords_3d):
    pose = self._coords_to_pose(sequence, coords_3d)
    energy = self.scorefxn(pose)
    return float(energy)


"Mapas de contacto"
def mapa_contacto_distancias(BB):
    BB_temp=BB[1::4]
    MC=[[0 for _ in range(len(BB_temp))]for _ in range(len(BB_temp)-1)]
    for i in range(len(BB_temp)-1):
        temporal= np.linalg.norm(BB_temp[(i+1):] - BB_temp[i], axis=1)
        MC[i][(i+1):]=temporal
    return MC

def mapa_contacto_binario(MC_dist, umbral=8.0):
    """
    Convierte una matriz de distancias en un mapa de contacto binario.

    Parámetros:
    - MC_dist: matriz de distancias (triangular o completa)
    - umbral: distancia máxima para considerar contacto (Å)

    Retorna:
    - matriz binaria de contactos (0 / 1)
    """

    MC_dist = np.array(MC_dist)
    MC_contacto = (MC_dist <= umbral).astype(int)
    return MC_contacto

def tm_score(x, y):
    """
    Calcula el TM-score según Zhang & Skolnick (2004)
    usando solo átomos Cα.
    """

    x = np.array(x)
    y = np.array(y)

    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    y_on_x = sup.get_transformed()

    # Distancias Cα
    distancias = np.linalg.norm(x - y_on_x, axis=1)
    dist_ca = distancias[1::4]

    n = len(dist_ca)

    if n <= 15:
        return 0.0  # definición estándar

    d0 = 1.24 * ((n - 15) ** (1/3)) - 1.8

    tm = np.sum(1.0 / (1.0 + (dist_ca / d0) ** 2)) / n
    return tm

def descriptores_seleccionados(secuencia,modelo):
    fastas = [['nombre',secuencia,'Chain','Chain']]
    userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
    if len(userDefinedOrder) != 20:
        userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    myAAorder = {
        'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
        'polarity': 'DENKRQHSGTAPYVMCWIFL',
        'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
        'userDefined': userDefinedOrder
    }
    myOrder = 'ACDEFGHIKLMNPQRSTVWY'
    kw = {'path': 'nada', 'order': myOrder, 'type': 'Protein'}
    cmd = modelo + '.' + modelo + '(fastas, **kw)'
    encodings = eval(cmd)
    return encodings[1][2:]


"descritores ESM2"
def ESM2_desc(secuencia,tokenizer,model_ESM2):
	inputs = tokenizer(secuencia, return_tensors="pt", padding=True, truncation=True)
	outputs = model_ESM2(**inputs)
	last_hidden_states = outputs.last_hidden_state
	x = last_hidden_states.detach()
	descriptores_ESM2= np.array([])
	for i in range(1,(len(x[0])-1)):
		descriptores_ESM2=np.concatenate((descriptores_ESM2, x[0][i].numpy()))
	return descriptores_ESM2
		


"determina los descriptores para una secuencia de aminoacidos segun el metodo seleccionado"
def descriptores(secuencia,tokenizer,model_ESM2):
    choices=['GAAC', 'EGAAC', 'CKSAAGP','GDPC','GTPC','BLOSUM62',
                  'CTDC', 'CTDT', 'CTDD']
    descriptores_salida = [None for _ in range(len(choices))] 
    iii=0
    for meot in choices:
        descriptores_salida[iii]=descriptores_seleccionados(secuencia,meot)
        iii+=1
    descriptores_salida.append(ESM2_desc(secuencia,tokenizer,model_ESM2))
    return descriptores_salida


"determina las metricas utilizadas en el fitness"
"se incluyen los descriptores fisico quimico"
def fitness_gdt_rmsd_mc_fisquim(x,y,MC_BB,sequence,corte,descriptor_ref,descriptor_temp):
        #rmsd
    x=np.array(x)
    y=np.array(y)
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rms = sup.get_rms()

    #gdt
    y_on_x = sup.get_transformed()
    distancias = np.linalg.norm(x - y_on_x, axis=1)
    distancias1=distancias
    gdt=0
    for i in range(len(corte)):
        gdt = gdt+np.count_nonzero(distancias1 <= corte[i])
    gdt=gdt/(len(corte)*len(distancias1))

    #RMSD MC
    mapa_distancias_MC = mapa_contacto_distancias(y)
    mapa_binario_MC = mapa_contacto_binario(mapa_distancias_MC)
    MC_similitud=rmsd_MC(MC_BB, mapa_binario_MC)


    divKl = calcular_divergencias(descriptor_ref, descriptor_temp)

    #tms
    #n=len(distancias_sal)
    #distancias1=np.array(distancias_sal)
    #d0=1.24*((n-15)*(1/3))-1.8
    #distancias1=distancias1/d0
    #distancias1=distancias1**2+1
    #tms=np.sum(1/distancias1)/n
    tms = tm_score(x,y)

    #energia
    energia_design = _calculate_design_energy(sequence,MC_BB)

    return rms, gdt, MC_similitud, divKl, tms, energia_design


"determina las metricas utilizadas en el fitness"
"no se incluyen los descriptores fisico quimico"
def fitness_gdt_rmsd_mc(x,y,MC_BB, sequence,corte):
    #RMSD
    x=np.array(x)
    y=np.array(y)
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rms = sup.get_rms()

    #gdt
    y_on_x = sup.get_transformed()
    distancias = np.linalg.norm(x - y_on_x, axis=1)
    distancias1=distancias
    gdt=0
    for i in range(len(corte)):
        gdt = gdt+np.count_nonzero(distancias1 <= corte[i])
    gdt=gdt/(len(corte)*len(distancias1))

    #RMSD MC
    mapa_distancias_MC = mapa_contacto_distancias(y)
    mapa_binario_MC = mapa_contacto_binario(mapa_distancias_MC)
    MC_similitud=rmsd_MC(MC_BB, mapa_binario_MC)

    #tms
    # n=len(distancias_sal)
    # distancias1=np.array(distancias_sal)
    # d0=1.24*((n-15)**(1/3))-1.8
    # distancias1=distancias1/d0
    # distancias1=distancias1**2+1
    # tms=np.sum(1/distancias1)/n
    tms = tm_score(x,y)

    #energia
    energia_design = _calculate_design_energy(sequence,MC_BB)

    return rms, gdt, MC_similitud, tms, energia_design

"agrega las metricas cuando se tienen descriptores"
def agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms):
    temporal_fitnes=(1/(1+rms))+gdt+(1/(1+MC_similitud))
    return temporal_fitnes+ (temporal_fitnes/3)/(1+np.exp((energia_desing+a)/b))+1/(1+np.mean(divKl))

"agrega las metricas cuando no se tienen descriptores"
def agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms):
    temporal_fitnes=(1/(1+rms))+gdt+(1/(1+MC_similitud))+tms
    return temporal_fitnes+ (temporal_fitnes/4)/(1+np.exp((energia_desing+a)/b))
