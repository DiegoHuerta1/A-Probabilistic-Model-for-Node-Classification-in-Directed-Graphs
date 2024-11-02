# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:10:21 2024

@author: diego
"""


import json
import pickle
import networkx as nx



'''
The data obtained from the scraping of the Math Genealogy Project is used.

A Networkx graph is created with this information.

The resulting graph has the indices of the mathematicians as the names of the nodes.
 Additionally, for each node, the following attributes (which can be None) are included:
- name
- thesis
- school
- country
- year
- subject

'''

# folder with the scrapping results
input_path = ".//scraping_results//"


# filenames of the input data
name_data_file = 'data.json'
name_edges_file = 'edges_graph.json'


# folder to save the nx graph
output_path = ".//nx_data//"


# name to save the graph
name_graph_output = 'genealogy_nx_graph.json'



# -------------------------------------------------------------------------------------------------------



# read data

# open json file
with open(input_path + name_data_file, 'r') as infile:
    data_dict = json.load(infile)
    
# el archivo tiene la informacion bajo la llave 'data'
datos_matematicos_lista = data_dict['data']


# transformar esta lista en un diccionario
# donde el id de cada matematico es la llave

# diccionario donde poner los datos
datos_matematicos = dict()

# iterar en los diccionarios de los matematicos
for info_matematico in datos_matematicos_lista:
    
    # tomar el id
    id_m = info_matematico['id']
    
    # agregar la entrada al diccionario
    # el id es la llave, toda la info es el valor
    datos_matematicos[id_m] = info_matematico
    
    
# -------------------------------------------------------------------------------------------------------

# read edges of the graph    
    

# open json
with open(input_path + name_edges_file, 'r') as infile:
    edges_graph = json.load(infile)
    

# tomar solo los edges
edges_list = edges_graph['edges']



# --------------------------------------------------------------------------------------------------------


# Creacion del grafo con toda la informacion
# es un grafo dirigido
grafo_matematicos = nx.DiGraph()

# poner todos los nodos, es decir, todos los matematicos
# iterar en la info de todos los matematicos
for id_matematico, info_matematico in datos_matematicos.items():
    
    # añadir el nodo, que es el id del matematico
    # junto con todos los atributos que tenga
    grafo_matematicos.add_node(id_matematico, 
                               name= info_matematico['name'],
                               thesis= info_matematico['thesis'],
                               school= info_matematico['school'],
                               country= info_matematico['country'],
                               year= str(info_matematico['year']),
                               subject= info_matematico['subject'])
    

# añadir todas las aristas, ya se tiene la lista
grafo_matematicos.add_edges_from(edges_list)


# guardar el grafo
with open(output_path + name_graph_output, 'wb') as f:
    pickle.dump(grafo_matematicos, f)
    

'''
Se carga como

with open(carpeta_datos + 'genealogy_nx_graph.json', 'rb') as f:
    G = pickle.load(f)

'''













