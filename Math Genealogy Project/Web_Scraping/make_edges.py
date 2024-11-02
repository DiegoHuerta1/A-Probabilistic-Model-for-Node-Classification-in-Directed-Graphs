''' Create edges from the data '''


import json
from tqdm import tqdm


# folder to save the data
output_path = "..\\Data\\scraping_results\\"


# names of the files to be produced
filename_edges_graph = "edges_graph.json"


'''
El grafo considera aristas dirigidas de los matematicos a sus estudiantes
'''


# open created data fetch.py
with open(output_path + 'data.json', 'r') as infile:
    data = json.load(infile)


# -----------------------------------------------------------------------------

# Create the edges of the graph

print("Create the edges of the graph")

# save the edges of the graph
graph_edges = []

# ir guardando los nodos
nodes = []

# iterar en todos los matematicos
for record in data['data']:

    # tomar su ID
    _id = record['id']

    # salvar su id, se tiene info para este mat
    nodes.append(_id)

    # añadir los correspondientes edges del grafo

    for student in record['students']:
        # añadir sus estudiantes, edges (matematico, estudiantes)
        graph_edges.append((_id, student))
    for advisor in record['advisors']:
        # añadir sus mentores, edges (advisor, matematico)
        graph_edges.append((advisor, _id))


# se terminan de agregar las aristas del grafo


# limpiar, quitar repetidos

graph_edges = [
    list(x) for x in sorted(list(set(graph_edges)))
]


print(f"\nEdges in the graph: {len(graph_edges)}")


# ---------------------------------------------------------------------------------


# una arista es valida si se tiene info de ambos extremos
# validar un conjunto de aristas

# hacer los nodos set
nodes = set(nodes)


def validate_edge_list(edges):

    # hacer la lista de aristas validadas
    validated_edges = []

    # iterar los edges
    for (source, target) in tqdm(edges):


        # edge malo por source
        if source not in nodes:
            print("Edge ({}, {}) is missing source".format(source, target))
            continue

        # edge malo por target
        if target not in nodes:
            print("Edge ({}, {}) is missing target".format(source, target))
            continue

        # edge malo por self loops
        if target == source:
            print("Edge ({}, {}) is a self loop".format(source, target))
            continue

        # si no es malo, guardar
        validated_edges.append((source, target))

    return validated_edges




print("\nValidate edges of the graph")
graph_edges = validate_edge_list(graph_edges)


print(f"\nEdges in the graph: {len(graph_edges)}")

# ---------------------------------------------------------------------------------

# write files

with open(output_path + filename_edges_graph, 'w') as outfile:
    json.dump({'edges': graph_edges}, outfile)
