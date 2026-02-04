import numpy as np
import re
from ..descproteins import GAAC,EGAAC,CKSAAGP,GDPC,GTPC,BLOSUM62,CTDC,CTDT,CTDD
from Bio.SVDSuperimposer import SVDSuperimposer
from ..EDA_tools.EDAtools import similitud_MC,entropia_descrip_val,entropia_descrip_prob, rmsd_MC, calcular_divergencias
from transformers import EsmTokenizer, EsmModel
import torch
import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta.core.pose import make_pose_from_sequence
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta import pose_from_sequence, get_fa_scorefxn

#Esta funcion es para crear el tipo Pose, que neceista Rosetta para calcular la energía: 
def _coords_to_pose(sequence, coords_3d):
    pose = pyrosetta.Pose()

    # esto crea el pose desde la secuencia (fa_standard)
    make_pose_from_sequence(pose, sequence, "fa_standard")

    coords_3d = np.asarray(coords_3d, dtype=float)
    assert coords_3d.shape == (len(sequence) * 4, 3)

    atom_idx = 0
    for i in range(1, pose.total_residue() + 1):
        for atom in ("N", "CA", "C", "O"):
            x, y, z = coords_3d[atom_idx]
            atomno = pose.residue(i).atom_index(atom)
            pose.set_xyz(AtomID(atomno, i), xyzVector_double_t(float(x), float(y), float(z)))
            atom_idx += 1

    return pose

#y este es la funcion, que dado el Pose , calcula la energía
def _calculate_design_energy(sequence):
    pose = pose_from_sequence(sequence)
    scorefxn = get_fa_scorefxn()   # score12 / ref2015
    energia = scorefxn(pose)
    print("Energía total:", energia)
    return energia


"Mapas de contacto"
def mapa_contacto_distancias(BB):
    BB_temp=BB[1::4]
    MC=[[0 for _ in range(len(BB_temp))]for _ in range(len(BB_temp)-1)]
    for i in range(len(BB_temp)-1):
        temporal= np.linalg.norm(BB_temp[(i+1):] - BB_temp[i], axis=1)
        MC[i][(i+1):]=temporal
    #print(MC)
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

"Mapas de contacto"
def mapa_contacto(BB):
    BB_temp=BB[1::4]
    MC=[[0 for _ in range(len(BB_temp))]for _ in range(len(BB_temp)-1)]
    for i in range(len(BB_temp)-1):
        temporal= np.linalg.norm(BB_temp[(i+1):] - BB_temp[i], axis=1)
        MC[i][(i+1):]=temporal
    return MC

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
def fitness_gdt_rmsd_mc_fisquim(x,y,sequence_b, sequence_a,MC_BB,corte,descriptor_ref,descriptor_temp):
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
    mapa_distancias_MC_Y = mapa_contacto_distancias(y)
    mapa_binario_MC_Y = mapa_contacto_binario(mapa_distancias_MC_Y)

    mapa_distancias_MC_X = mapa_contacto_distancias(x)
    mapa_binario_MC_X = mapa_contacto_binario(mapa_distancias_MC_X)

    MC_similitud=rmsd_MC(mapa_binario_MC_X, mapa_binario_MC_Y)


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
    energia_design_a = _calculate_design_energy(sequence_a)
    energia_design_b = _calculate_design_energy(sequence_b)

    return rms, gdt, MC_similitud, divKl, tms, energia_design_a, energia_design_b


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
    energia_design_a = _calculate_design_energy(sequence) 
    energia_design_b = _calculate_design_energy(sequence) #esta bien 

    return rms, gdt, MC_similitud, tms, energia_design_a, energia_design_b

"agrega las metricas cuando se tienen descriptores"
def agrega_rmsd_gdt_E_MC_divKl(MC_similitud,rms,gdt,divKl,energia_desing_a,energia_desing_b,b,tms):
    f1=(1/(1+rms))+gdt+(1/(1+MC_similitud))+tms
    diffE = abs(energia_desing_a-energia_desing_b)
    f2=(f1/4)*(1+np.exp(-diffE/b))
    f3=f3_descriptores(divKl,2)
    return f1+f2+f3

"agrega las metricas cuando no se tienen descriptores"
def agrega_rmsd_gdt_E_MC(MC_similitud,rms,gdt,energia_desing_a,energia_desing_b,b,tms):
    f1=(1/(1+rms))+gdt+(1/(1+MC_similitud))+tms
    diffE = abs(energia_desing_a-energia_desing_b)
    f2=(f1/4)*(1+np.exp(-diffE/b))
    return f1+f2


def f3_descriptores(divKl, delta=1.0):
    divKl = np.asarray(divKl, dtype=float)
    divKl = divKl[np.isfinite(divKl)]
    if divKl.size == 0:
        # si no hay valores válidos, aplicar por default
        divKl = np.array([2.5], dtype=float)

    m = divKl.size
    mean_div = float(np.sum(divKl) / m)

    f3 = (2.0 * delta) / (1.0 + mean_div)
    return float(f3)
