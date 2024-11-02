# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:01:24 2024

@author: diego
"""


import json
import pickle
import networkx as nx
import random
import xml.etree.ElementTree as ET


'''
Toma el grafo de nx creado
Agrega las prediccioens de diferentes metodos
Crea un subgrafo conexo para hacer visualizaciones
'''



# carpeta donde guardar los subgrafos
output_path = ".//graphs_with_predictions_ml//"


# carpeta de donde se leen los datos
input_path =  "..//Data//Probabilistic_Inference_data//"



# leer el grafo
with open(input_path + 'data_graph.json', 'rb') as f:
    complete_graph = pickle.load(f)
    
    
    
# -------------------------------------------------------------------------------

# Tomar predicciones
    
# delimitar el nombre de la carpeta donde se tienen las predicciones
predictions_path =   "..//Predictions//"

# para cada metodo delimitar el nombre del archivo de predicciones
nombre_archivo_predicciones = {
    "Naive Bayes" : "Naive_Bayes.pkl",
    "BERT": 'BERT.pkl',
    "GCN": 'GCN.pkl',
    "Label Propagation": 'Label_Propagation.pkl',
    "Our_model_ML": 'probabilistic_graph_model_ML.pkl',
    "Our_model_MAP": 'probabilistic_graph_model_MAP.pkl'
}



# funcion para leer un archivo pickle
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


# por cada metodo leer el diccionario de predicciones
diccionarios_predicciones = {
    metodo: load_pickle_file(predictions_path + nombre_predicciones)
    for metodo, nombre_predicciones in nombre_archivo_predicciones.items()
}



# -----------------------------------------------------------------------------


# nombre del archivo con la division de los nodos
nombre_division = "..//Data//node_division.json"


# cargar division de nodos
with open(nombre_division, 'rb') as f:
    division_nodos = json.load(f)
    
    
# obtener los conjuntos de nodos, hacerlo conjunto
nodos_train = set(division_nodos['nodos_train'])
nodos_val = set(division_nodos['nodos_val'])
nodos_test = set(division_nodos['nodos_test'])
nodos_useless = set(division_nodos['nodos_useless'])



# funcion que da el estado de un nodo
def obtener_estado_nodo(v):
    
    # ver a donde pertenece
    if v in nodos_train:
        estado_v = "train"
        
    elif v in nodos_val:
        estado_v = "val"
    
    elif v in nodos_test:
        estado_v = "test"
        
    elif v in nodos_useless:
        estado_v = "useless"
        
    # si no pertenece a ninguna es porque falta thesis
    else:
        estado_v = "FALTA THESIS"
        
    return estado_v
        

# -------------------------------------------------------------------------------


# helper functions


# validar que un string solo tenga caracteres validos para xmls
def validate_unicode_for_xml(text, replacement='?'):

    # intentar pasarlo a xml, si si se puede todo bien
    try:
        # Try to create an ElementTree with the given text
        ET.fromstring('<root>' + text + '</root>')
        return text
    
    # si no se puede, se debe de modificar
    except ET.ParseError:
        
        
        # se quitan los caracteres que causen problema
        cleaned_text = ''.join(c if c.isprintable() and ord(c) < 0x10FFFF else replacement for c in text)
        
        
        return cleaned_text




# modifies a graph so it can be saved as a graphml
def adapt_graph_to_graphml(G):
    
    # copy the graph
    G_new = G.copy()
    
    
    # iterate on the nodes
    for node_id, info_matematico in G_new.nodes(data=True):
        
        
        # get new node features
        thesis_matematico = info_matematico['thesis'] if info_matematico['thesis'] is not None else "None"
        
        
        # validate them
        thesis_matematico = validate_unicode_for_xml(thesis_matematico)
        
        
        # update node features
        info_matematico['thesis'] = thesis_matematico
        
    # return new graph
    return G_new



def agregar_predicciones_grafo(G):
    '''
    Dado un grafo G
    Agregar las prediccioens de subject para los nodos
    '''
    
    # iterar en los nodos
    for v in G.nodes():
        
        # agregar una prediccion por cada metodo
        
        # iterar en los metodos
        for metodo in nombre_archivo_predicciones.keys():
            
            # poner la prediccion para este metodo
            # si es que se tiene thesis
            
            # si tiene tesis poner la prediccion que se hace
            if G.nodes[v]["thesis"] != "None":
                G.nodes[v][metodo] = str(diccionarios_predicciones[metodo][v])
                
            # si no tiene tesis indicarlo
            else:
                G.nodes[v][metodo] = "FALTA THESIS"
                
            
        # end for iterar metodos
        
    # end for iterar nodos
    return G
    


# agregar estados
def agregar_division_nodos(G):
    '''
    Dado un grafo G
    Agregar la division de cada nodo
    '''

    # iterar en los nodos
    for v in G.nodes():
        
        # poner en que estado esta
        G.nodes[v]['division'] = obtener_estado_nodo(v)
        
    # devovler
    return G

# -------------------------------------------------------------------------------

# Get a subgraph of the graph or the tree


# funcion para crear el subgrafo deseado
# tipo \in {tree, graph}
def crear_salvar_subgrafo(nodo_inicial = None, tamaño_maximo = 100):
    
    
    # tomar el grafo
    G = complete_graph.copy()

    # quitarle direccion
    grafo = G.to_undirected()    
    
    # Inicializar un conjunto para almacenar los nodos visitados
    nodos_visitados = set()
    
    # si no hay nodo inicial ponerlo aleatorio
    # tal qe este en el componente gigante
    if nodo_inicial is None:
        giant_component = max(nx.connected_components(grafo), key=len)
        nodo_inicial = random.choice(list(giant_component))

    
    # Inicializar una cola para el BFS
    cola = [nodo_inicial]

    # Realizar BFS
    while cola:
        # Sacar el primer nodo de la cola
        nodo_actual = cola.pop(0)

        # Agregar el nodo actual a los nodos visitadosa
        nodos_visitados.add(nodo_actual)

        # Si hemos alcanzado el tamaño objetivo, terminar
        if len(nodos_visitados) >= tamaño_maximo:
            break

        # Agregar los vecinos no visitados del nodo actual a la cola
        vecinos_no_visitados = [vecino for vecino in grafo.neighbors(nodo_actual) 
                                if vecino not in nodos_visitados]
        cola.extend(vecinos_no_visitados)


    # se alcanzo el tamaño maximo,
    # o se termino el componente conexo
    
    # Crear un subgrafo inducido con los nodos visitados
    subgrafo = G.subgraph(nodos_visitados)
    
    
    # ver el tamaño que tiene
    tamaño_subgrafo = subgrafo.number_of_nodes()
    
    # crear el nombre con el que se guarda
    name_guardar = f"Subgraph_{nodo_inicial}_{tamaño_subgrafo}.graphml"
    
    # adaptar para que pueda ser guardado como graphml
    subgrafo = adapt_graph_to_graphml(subgrafo)
    
    # agregar las predicciones
    subgrafo = agregar_predicciones_grafo(subgrafo)
    
    # agregar la informacion de la division de nodos
    subgrafo = agregar_division_nodos(subgrafo)
    
    
    # guardar el grafo
    nx.write_graphml(subgrafo, output_path + name_guardar)
    
    # indicar lo creado
    print("-"*50)
    print("Se crea un subgrafo inducido del grafo completo")
    print(f"Se contiene al nodo {nodo_inicial}")
    print(f"El subgrafo es de tamaño {tamaño_subgrafo}")
    print("-"*50)


    return 1


# -----------------------------------------------------------------------------------

if __name__=='__main__':
    
    # aleatorio
    crear_salvar_subgrafo(None, 100)
    crear_salvar_subgrafo(None, 500)
    
    # arizmendi: 202780
    crear_salvar_subgrafo(202780, 100)
    crear_salvar_subgrafo(202780, 500)

    # Marina Meila: 87374
    crear_salvar_subgrafo(87374, 100)
    crear_salvar_subgrafo(87374, 500)
    


