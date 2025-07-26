# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:39:34 2024

@author: diego
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from scipy.stats import multinomial
from sklearn.pipeline import Pipeline
import pickle
import json
import random
from collections import Counter
import distributions
from tqdm import tqdm



class probabilistic_graph_model:
    '''
    Model from: A Probabilistic Model for Node Classification in Directed Graphs
    Considers text atribute data, therefore,
    the functions omega are estimated using a Naive Bayes approach
    '''

    # constructor
    def __init__(self, graph,
                 name_atributes_x = 'x',
                 name_label_y = 'y',
                 name_division = 'division',
                 decode_label = None):
        '''
        graph - Networkx graph
        name_atributes_x - atribute of nodes denoting their features
        name_label_y -     atribute of nodes denoting their labels
                            the labels should be {0, 2, ..., K-1} for some K
        name_division    - atribute of nodes denoting its division, options:
                            {train, val, test, useless}
        decode_label    - dictionary mapping label indices to any value
        '''

        # para graficas
        sns.set_theme()

        # set atributes of the class
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()
        self.name_atributes_x = name_atributes_x
        self.name_label_y = name_label_y
        self.name_division = name_division

        # obtain the division of nodes
        train_nodes = np.array([v for v, info_v in
                                self.graph.nodes(data=True)
                                if info_v[self.name_division] == 'train'])
        val_nodes = np.array([v for v, info_v in
                              self.graph.nodes(data=True)
                              if info_v[self.name_division] == 'val'])
        test_nodes = np.array([v for v, info_v in
                              self.graph.nodes(data=True)
                              if info_v[self.name_division] == 'test'])
        useless_nodes = np.array([v for v, info_v in
                                  self.graph.nodes(data=True)
                                  if info_v[self.name_division] == 'useless'])
        # assert it is a partition (kind of)
        assert len(train_nodes) + len(val_nodes) + len(test_nodes) + len(useless_nodes) == self.n
        # set atributes
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.useless_nodes = useless_nodes
        self.set_train_nodes = set(train_nodes)
        self.set_no_train_nodes = np.array(list(set(self.graph.nodes()) - self.set_train_nodes))
        self.list_no_train_nodes = list(self.set_no_train_nodes)

        # assert that labels are indices 0,...,K-1
        labels_train_nodes = [info_v[self.name_label_y] for v, info_v
                              in self.graph.nodes(data=True)
                              if v in self.train_nodes]
        unique_labels = np.array(list(set(labels_train_nodes)))
        unique_labels.sort()
        K = max(unique_labels) + 1
        assert (unique_labels == np.arange(K)).all()
        # set atributes
        self.K = K

        # if decode_label not provided, the its the identity
        if decode_label is None:
            decode_label = {y:y for y in unique_labels}
        # set atributes
        self.decode_label = decode_label

    # end constructor -----------------------------------------


    def print_info(self):
        '''
        Print general information of the model
        '''
        print("-"*100)
        print(f"Number of nodes: {self.n}")
        print(f"Number of edges: {self.m}")
        print(f"Number of classes: {self.K}\n")
        print(f"Training nodes:   {len(self.train_nodes)} ({100*len(self.train_nodes)/self.n:.2f}%)")
        print(f"Validation nodes: {len(self.val_nodes)} ({100*len(self.val_nodes)/self.n:.2f}%)")
        print(f"Testing nodes:    {len(self.test_nodes)} ({100*len(self.test_nodes)/self.n:.2f}%)")
        print(f"Useless nodes:    {len(self.useless_nodes)} ({100*len(self.useless_nodes)/self.n:.2f}%)")
        print("-"*100)
    # end print info


    # -------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Statistical tests

    # test for zero truncated power law
    def analize_zero_truncated_pw(self, mode = "in",
                                  labels_check = None,
                                  D_max = 1000,
                                  number_cells = 10,
                                  show_ind_results = False,
                                  devolver = False):
        '''
        Goodness of fit test to check the proposed distribution
        for in/out degree

        mode - in/out to check in/out degree distributions
        D_max - upper bound on the degree
        number_cells - number of cells for the chi square test
        labels_check - set of labels to check

        The purpose is to check statistical validation,
        not to learn parameters.
        Therefore, all nodes are used, not only training nodes
        '''

        print("Analize zero truncated power law distribution")
        print(f"For the {mode} distribution\n")

        # si no se especifican las labels, ver todas
        if labels_check is None:
            labels_check = np.arange(self.K)


        # por cada label que se quiere analizar
        # tomar la distribucion de in/out
        samples = {i:[] for i in labels_check}

        # iterar en todos los nodos con sus datos
        for v, info_v in self.graph.nodes(data=True):

            # que no sea nodo useless
            if v in self.useless_nodes:
                continue

            # identificar su label
            label_nodo = info_v[self.name_label_y]

            # solo si es de las labels de interes
            if label_nodo not in labels_check:
                continue

            # agregar su in/out degree segun lo que se quiera
            if mode == "in":
                samples[label_nodo].append(self.graph.in_degree(v))
            elif mode == "out":
                samples[label_nodo].append(self.graph.out_degree(v))

        # guardar pvalue de cada label a checar
        pvalues_labels = []

        # por cada label de interes
        for label in labels_check:
            if show_ind_results:
                print(f"\nLabel: {label} ({self.decode_label[label]})")
            # hacer su analisis
            pvalue_label = distributions.analize_zero_truncated_pw(sample = samples[label],
                                                                   D_max = D_max,
                                                                   k_chi2 = number_cells,
                                                                   ver = show_ind_results)
            pvalues_labels.append(pvalue_label)


        # ya se tienen los pvalues de todos
        # mostrar los resultados

        # esto solo si se evaluo mas de una
        if len(labels_check) > 1:

            # transformar
            resultados = ["No" if pvalue < 0.05 else "Yes" for pvalue in pvalues_labels]
            # ver frecuencias
            frecuencias_resultados = Counter(resultados)

            # ver
            plt.bar(frecuencias_resultados.keys(), frecuencias_resultados.values())
            plt.title("Number of samples that follow the distribution")
            plt.show()

            print("")
            print(f"Yes : {frecuencias_resultados['Yes']} ({100*frecuencias_resultados['Yes']/len(resultados)})")
            print(f"No : {frecuencias_resultados['No']} ({100*frecuencias_resultados['No']/len(resultados)})")

            # ver casos no favorables
            print("\nUnsuccessful labels:")
            for idx, label in enumerate(labels_check):
                # si no es favorable
                if resultados[idx] == "No":
                    print(f"Label: {label} ({self.decode_label[label]})")


        if devolver:
            # devolver pvalues y resultados
            return pvalues_labels, frecuencias_resultados



    # test for zero truncated power law
    def analize_zero_discrete_lognormal(self, mode = "in",
                                        labels_check = None,
                                        initial_point = [0.5, 0.5],
                                        D_max = 1000,
                                        number_cells = 10,
                                        show_ind_results = False,
                                        devolver = False):
        '''
        Goodness of fit test to check the proposed distribution
        for in/out degree

        mode - in/out to check in/out degree distributions
        D_max - upper bound on the degree
        number_cells - number of cells for the chi square test
        labels_check - set of labels to check

        The purpose is to check statistical validation,
        not to learn parameters.
        Therefore, all nodes are used, not only training nodes
        '''

        print("Analize zero discrete lognormal distribution")
        print(f"For the {mode} distribution\n")

        # si no se especifican las labels, ver todas
        if labels_check is None:
            labels_check = np.arange(self.K)

        # por cada label que se quiere analizar
        # tomar la distribucion de in/out
        samples = {i:[] for i in labels_check}

        # iterar en todos los nodos con sus datos
        for v, info_v in self.graph.nodes(data=True):

            # que no sea nodo useless
            if v in self.useless_nodes:
                continue

            # identificar su label
            label_nodo = info_v[self.name_label_y]

            # solo si es de las labels de interes
            if label_nodo not in labels_check:
                continue

            # agregar su in/out degree segun lo que se quiera
            if mode == "in":
                samples[label_nodo].append(self.graph.in_degree(v))
            elif mode == "out":
                samples[label_nodo].append(self.graph.out_degree(v))

        # guardar pvalue de cada label a checar
        pvalues_labels = []

        # por cada label de interes
        for label in labels_check:
            if show_ind_results:
                print(f"\nLabel: {label} ({self.decode_label[label]})")
            # hacer su analisis
            pvalue_label = distributions.analize_zero_discrete_lognormal(sample = samples[label],
                                                                         initial_point = initial_point,
                                                                         D_max = D_max,
                                                                         k_chi2 = number_cells,
                                                                         ver = show_ind_results)
            pvalues_labels.append(pvalue_label)


        # ya se tienen los pvalues de todos
        # mostrar los resultados

        # esto solo si se evaluo mas de una
        if len(labels_check) > 1:

            # transformar
            resultados = ["No" if pvalue < 0.05 else "Yes" for pvalue in pvalues_labels]
            # ver frecuencias
            frecuencias_resultados = Counter(resultados)

            # ver
            plt.bar(frecuencias_resultados.keys(), frecuencias_resultados.values())
            plt.title("Number of samples that follow the distribution")
            plt.show()

            print("")
            print(f"Yes : {frecuencias_resultados['Yes']} ({100*frecuencias_resultados['Yes']/len(resultados)})")
            print(f"No : {frecuencias_resultados['No']} ({100*frecuencias_resultados['No']/len(resultados)})")

            # ver casos no favorables
            print("\nUnsuccessful labels:")
            for idx, label in enumerate(labels_check):
                # si no es favorable
                if resultados[idx] == "No":
                    print(f"Label: {label} ({self.decode_label[label]})")


        if devolver:
            # devolver pvalues y resultados
            return pvalues_labels, frecuencias_resultados



    # -------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Estimation of parameters

    def estimate_matrix_log_psi(self, config, folder_parameters):
        '''
        Estimate a matrix with log evaluations of psi
        this denotes the distribution of in-degree acording to label
        maitrx_log_psi_{i, j} = log psi_i(j) = log P(D^in = j | Y = i)
        '''

        # segun la forma de diestribucion que se quiera, se debe de estimar

        # first, create a diccionary of samples,
        # for each label, create a sample of in-degree (of traininng nodes)
        samples_in_degree_label = {i:[] for i in range(self.K)}

        # iterar en todos los nodos con sus datos
        for v, info_v in self.graph.nodes(data=True):

            # SOLO CONSIDERAR NODOS EN TRAIN
            # si no es nodo de train, continuar
            if v in self.set_no_train_nodes:
                continue

            # identificar su label
            label_nodo = info_v[self.name_label_y]
            # agregar su in degree
            samples_in_degree_label[label_nodo].append(self.graph.in_degree(v))

        # ahora llenar la matriz
        # iniciar vacia
        matrix_log_psi = np.zeros((self.K, config["D_in_max"]+1))

        # llenar para cada clase
        for i in range(self.K):
            # tomar el sample correspondiente
            sample_label_i = np.array(samples_in_degree_label[i])

            # hacer la estimacion de distribucion segun lo que se quiera

            # frecuencias y smoothing
            if config["psi_dist"] == "additive_smoothing":

                # llamar ala funcion qeu da las probabilidades
                probabilidades = distributions.frecuency_probabilities(sample_label_i,
                                                                       smoothing_parameter = config["alpha_psi"],
                                                                       max_value = config["D_in_max"])
                # poner en la matriz con logaritmo
                matrix_log_psi[i] = np.log(probabilidades)


            # truncated power law y cero
            elif config["psi_dist"] == "zero_truncated_power_law":

                # estimar los parametros de la distribucion
                parametros_i = distributions.estimate_zero_truncated_pw(sample_label_i,
                                                                        D_max = config["D_in_max"]+1)
                # separar
                beta_i = parametros_i['beta']
                kappa_i = parametros_i['kappa']
                lamda_i = parametros_i['lamda']

                # calcular las probabilidades en log
                probabilidades = distributions.log_pmf_zero_truncated_pw(d = np.arange(config["D_in_max"]+1),
                                                                         beta = beta_i,
                                                                         kappa = kappa_i,
                                                                         lamda = lamda_i,
                                                                         D_max = config["D_in_max"]+1)
                # poner en la matriz
                matrix_log_psi[i] = probabilidades

            # lognormal y cero
            elif config["psi_dist"] == "zero_lognormal":

                # estimar los parametros
                parametros_i = distributions.estimate_zero_discrete_lognormal(sample_label_i,
                                                                              D_max = config["D_in_max"]+1)

                # separar
                beta_i = parametros_i['beta']
                mu_i = parametros_i['mu']
                sigma_i = parametros_i['sigma']

                # calcular las log probabilidades
                probabilidades = distributions.log_pmf_zero_discrete_lognormal(d = np.arange(config["D_in_max"]+1),
                                                                               beta = beta_i,
                                                                               mu = mu_i,
                                                                               sigma = sigma_i,
                                                                               D_max = config["D_in_max"]+1)

                # poner en la matriz
                matrix_log_psi[i] = probabilidades

            # metodo no valido
            else:
                raise Exception("Psi distribution not valid")

        # salvar
        np.save(folder_parameters + 'matrix_log_psi.npy', matrix_log_psi)

        return matrix_log_psi


        # end estimate matrix_log_psi

    def estimate_matrix_log_phi(self, config, folder_parameters):
        '''
        Estimate a matrix with log evaluations of phi
        this denotes the distribution of out-degree acording to label
        maitrx_log_phi_{i, j} = log phi_i(j) = log P(D^out = j | Y = i)
        '''

        # segun la forma de diestribucion que se quiera, se debe de estimar

        # first, create a diccionary of samples,
        # for each label, create a sample of out-degree (of traininng nodes)
        samples_out_degree_label = {i:[] for i in range(self.K)}

        # iterar en todos los nodos con sus datos
        for v, info_v in self.graph.nodes(data=True):

            # SOLO CONSIDERAR NODOS EN TRAIN
            # si no es nodo de train, continuar
            if v in self.set_no_train_nodes:
                continue

            # identificar su label
            label_nodo = info_v[self.name_label_y]
            # agregar su out degree
            samples_out_degree_label[label_nodo].append(self.graph.out_degree(v))

        # ahora llenar la matriz
        # iniciar vacia
        matrix_log_phi = np.zeros((self.K, config["D_out_max"]+1))

        # llenar para cada clase
        for i in range(self.K):
            # tomar el sample correspondiente
            sample_label_i = np.array(samples_out_degree_label[i])

            # hacer la estimacion de distribucion segun lo que se quiera

            # frecuencias y smoothing
            if config["phi_dist"] == "additive_smoothing":

                # llamar ala funcion qeu da las probabilidades
                probabilidades = distributions.frecuency_probabilities(sample_label_i,
                                                                       smoothing_parameter = config["alpha_phi"],
                                                                       max_value = config["D_out_max"])
                # poner en la matriz con logaritmo
                matrix_log_phi[i] = np.log(probabilidades)


            # truncated power law y cero
            elif config["phi_dist"] == "zero_truncated_power_law":

                # estimar los parametros de la distribucion
                parametros_i = distributions.estimate_zero_truncated_pw(sample_label_i,
                                                                        D_max = config["D_out_max"]+1)
                # separar
                beta_i = parametros_i['beta']
                kappa_i = parametros_i['kappa']
                lamda_i = parametros_i['lamda']

                # calcular las probabilidades en log
                probabilidades = distributions.log_pmf_zero_truncated_pw(d = np.arange(config["D_out_max"]+1),
                                                                         beta = beta_i,
                                                                         kappa = kappa_i,
                                                                         lamda = lamda_i,
                                                                         D_max = config["D_out_max"]+1)
                # poner en la matriz
                matrix_log_phi[i] = probabilidades

            # lognormal y cero
            elif config["phi_dist"] == "zero_lognormal":

                # estimar los parametros
                parametros_i = distributions.estimate_zero_discrete_lognormal(sample_label_i,
                                                                              D_max = config["D_in_max"]+1)

                # separar
                beta_i = parametros_i['beta']
                mu_i = parametros_i['mu']
                sigma_i = parametros_i['sigma']

                # calcular las log probabilidades
                probabilidades = distributions.log_pmf_zero_discrete_lognormal(d = np.arange(config["D_in_max"]+1),
                                                                               beta = beta_i,
                                                                               mu = mu_i,
                                                                               sigma = sigma_i,
                                                                               D_max = config["D_in_max"]+1)

                # poner en la matriz
                matrix_log_phi[i] = probabilidades

            # metodo no valido
            else:
                raise Exception("Phi distribution not valid")

        # salvar
        np.save(folder_parameters + 'matrix_log_phi.npy', matrix_log_phi)

        return matrix_log_phi


    def estimate_NB_text(self, config, folder_parameters):
        '''
        Estimate the naibe bayes that is used to compute the probabilities of text
        that is, compute w_i(text)
        Naibe bayes is used as it follows almost the same probabilistic rules
        (we are considering text atributes in the model)
        Naive Bayes is used to compute: P(X = x | Y = i)
        '''

        # para entrenarl el NB, poner texto y label de todos los datos en train
        X_nb = []
        y_nb = []

        # iterar en los nodos
        for v, info_v in self.graph.nodes(data=True):

            # SOLO CONSIDERAR NODOS EN TRAIN
            # si no es nodo de train, continuar
            if v in self.set_no_train_nodes:
                continue

            # tomar el atributo (texto)
            X_nb.append(info_v[self.name_atributes_x])
            # tomar el label
            y_nb.append(info_v[self.name_label_y])


        # hacer el pipeline, vectorizacion y despues naibe bayes

        # la eleccion de vectorizador es un hyperparametro
        opciones_vectorizar = {"count": CountVectorizer, "tfidf": TfidfVectorizer}
        vectorizador = opciones_vectorizar[config["Vectorizer"]]


        # pasos del pipeline
        # varias selecciones son hyperparametros
        NB_steps = [('vectorizer', vectorizador(stop_words = 'english',
                                                min_df = config["Min_df"],
                                                max_df = config["Max_df"],
                                                ngram_range = config["Ngram_range"],
                                                max_features = config["Max_features"])),
                    ('NB', MultinomialNB(alpha = config["alpha_omega"]))]


        # crear el pipeline
        NB_pipeline = Pipeline(NB_steps)
        # entrenar el nb
        NB_pipeline.fit(X_nb, y_nb)
        # salvar pipeline
        with open(folder_parameters + 'NB_pipeline.pkl','wb') as f:
            pickle.dump(NB_pipeline,f)

        return NB_pipeline
    # end estimate log_pi


    def estimate_Theta_Xi(self, config, folder_parameters):
        '''
        Estimate the Theta and Xi matrices
        Theta_{i,j} = P(Y = j | predecesor = i)
        Xi_{i,j} =    P(Y = j | successor = i)
        '''


        # primero hacer una matriz de cuentas
        # n_{i,j} = numero de aristas que van de un nodo con label i a un nodo con label j

        # iniciar matriz en ceros
        matriz_cuentas = np.zeros((self.K, self.K), int)

        # iterar en las aristas
        for nodo_origen, nodo_destino in self.graph.edges():

            # SOLO CONSIDERAR ARISTAS QUE CONECTEN DOS NODOS EN TRAIN
            if nodo_origen in self.set_no_train_nodes or nodo_destino in self.set_no_train_nodes:
                continue

            # tomar los labels
            label_origen =  self.graph.nodes[nodo_origen][self.name_label_y]
            label_destino =  self.graph.nodes[nodo_destino][self.name_label_y]

            # agregar uno a la matriz
            matriz_cuentas[label_origen, label_destino] += 1


        # obtener matriz Theta -----------------------------------

        # añadir smoothing parameter
        cuentas_smoothing_Theta = matriz_cuentas + config["alhpa_Theta"]

        # dividir las filas de las cuentas sobre su suma para que sea probabilidad
        # se normaliza cada fila para que sume uno
        Theta = cuentas_smoothing_Theta / cuentas_smoothing_Theta.sum(axis=1)[:, np.newaxis]

        # salvar
        np.save(folder_parameters + 'Theta.npy', Theta)

        # obtener matriz Xi ------------------------------------------

        # añadir smoothing parameter
        cuentas_smoothing_Xi = matriz_cuentas + config["alpha_Xi"]

        # dividir las columnas de las cuentas sobre su suma para que sea probabilidad
        # se normaliza cada columna para que sume uno, despues se transpone
        Xi = (cuentas_smoothing_Xi / cuentas_smoothing_Xi.sum(axis=0)).T

        # salvar
        np.save(folder_parameters + 'Xi.npy', Xi)

        return Theta, Xi
    # end estimate Theta Xi


    def estimate_log_pi(self, config, folder_parameters):
        '''
        Estimate the log of pi, where
        pi_i = P(Y = i)
        '''

        # tomar las frecuencias de todos los labels
        # iniciar vacio
        frecuencias_labels = np.zeros(self.K)

        # iterar en los nodos
        for v, info_v in self.graph.nodes(data=True):

            # SOLO CONSIDERAR NODOS EN TRAIN

            # si no es nodo de train, continuar
            if v in self.set_no_train_nodes:
                continue

            # tomar el indice del label
            indice_v = info_v[self.name_label_y]
            # añadir al vector
            frecuencias_labels[indice_v] += 1

        # añadir el smoothing parameter
        frecuencias_labels = frecuencias_labels + config["alpha_pi"]
        # convertir las frecuencias en probabilidades
        probabilidades_labels = frecuencias_labels/frecuencias_labels.sum()
        # sacar logaritmo para tener el vector deseado
        log_pi = np.log(probabilidades_labels)
        # salvar
        np.save(folder_parameters + 'log_pi.npy', log_pi)

        return log_pi
    # end estimate log_pi


    def estimate_parameters(self, config, folder_parameters = "./"):
        '''
        Estimate all parameters of the model and save them as atributes of the object
        config - dictionary with hyperparameters
        folder_parameters - folder to save the parameters
        '''

        # call individual functions
        log_pi = self.estimate_log_pi(config, folder_parameters)
        Theta, Xi = self.estimate_Theta_Xi(config, folder_parameters)
        matrix_log_phi = self.estimate_matrix_log_phi(config, folder_parameters)
        matrix_log_psi = self.estimate_matrix_log_psi(config, folder_parameters)
        NB_pipeline = self.estimate_NB_text(config, folder_parameters)

        # poner en un dict
        parameters = {
            "log_pi": log_pi,
            "Theta": Theta,
            "Xi": Xi,
            "matrix_log_phi": matrix_log_phi,
            "matrix_log_psi": matrix_log_psi,
            "NB_pipeline": NB_pipeline,
        }
        self.parameters = parameters
    # end estimate parameters

    # -------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Preparar inferencia

    # poner proba de texto para nodos no train
    def precomputar_log_funcion_omega(self):
        '''
        Para cada nodo que no se de entrenamiento
        precomputar un vector con las cantidades
        log w_i(x) = log P(X = x | Y = i)
        para todos los i
        '''

        # tomar la nb pipeline
        NB_pipeline = self.parameters["NB_pipeline"]


        # para cada nodo que no sea de train
        # tomar el texto de atributo
        textos_nodos_no_train = [self.graph.nodes[v][self.name_atributes_x]
                                 for v in self.list_no_train_nodes]

        # vectorizar todos los textos
        textos_vectorizados = NB_pipeline['vectorizer'].transform(textos_nodos_no_train)

        # para cada texto, tomar las joint log proba para cada subject i
        # es decir: log P(x|i) + log P(i)
        # where log P(i) is the class prior probability
        log_joint_probas = NB_pipeline['NB'].predict_joint_log_proba(textos_vectorizados)

        # restar las log prior log P(i) para quedarnos solo con log P(x|i)
        log_conditional_probas = log_joint_probas - NB_pipeline['NB'].class_log_prior_

        # poner el valor como atributos de nodos

        # iterar los nododos que no son train
        for idx_v, v in enumerate(self.list_no_train_nodes):
            # ponerlo como un argumento del nodo
            self.graph.nodes[v]['log_omega_evaluated'] = log_conditional_probas[idx_v, :]



    # precomputar grados de nodos para no calcularlos
    def precomputar_grados(self):
        '''
        Para cada nodo que no sea de train, precomputar:
            Out degree
            In degree
        '''
        # iterar los nododos que no son train
        for v in self.set_no_train_nodes:
            # poner valores
            self.graph.nodes[v]['out_degree'] = self.graph.out_degree(v)
            self.graph.nodes[v]['in_degree'] = self.graph.in_degree(v)



    # hacer las primeras predicciones para nodos no train
    # para despues inferir con las iteraciones
    def hacer_predicciones_iteracion_0(self):
        '''
        Hace predicciones de label para cada nodo que no sea train
        se usa unicamente el texto atributo
        '''


        # si se quiere usar el vector para determinar el primer valor
        if self.config["method_iteration_0"] == "text":


            # tomar todos los textos de los nodos que no son de entrenamiento
            textos_nodos = np.array([self.graph.nodes[v][self.name_atributes_x]
                                     for v in self.list_no_train_nodes])
            # tomar el NB
            NB_pipeline = self.parameters["NB_pipeline"]
            # hacer predicciones en estos textos
            predicciones_iteracion_0 = NB_pipeline.predict(textos_nodos)

            # iterar en los nodos no train
            for idx_v, v in enumerate(self.list_no_train_nodes):

                # poner la prediccion correspondiente
                # poner como prediccion de mle y de map
                self.graph.nodes[v]['prediccion_mle_0'] = predicciones_iteracion_0[idx_v]
                self.graph.nodes[v]['prediccion_map_0'] = predicciones_iteracion_0[idx_v]


        # si se quiere hacer la iteracion 0 al azar
        elif self.config["method_iteration_0"]== "random":

            # iterar en los nodos no train
            for v in self.set_no_train_nodes:
                # tomar un label al azar
                label_azar = np.random.randint(self.K)
                # poner como prediccion de mle y de map
                self.graph.nodes[v]['prediccion_mle_0'] = label_azar
                self.graph.nodes[v]['prediccion_map_0'] = label_azar


        # si se quieren usar nodos cercanos
        elif self.config["method_iteration_0"]== "near":

            # asignar los nodos con una funcion especial
            self.asignar_subject_nodo_cercano()

        # otro metodo no valido
        else:
            raise Exception("Metodo para iteracion 0 no valido")

    # -------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Heuristico para la iteracion 0

    def asignar_subject_nodo_cercano(self):
        '''
        Asigna el label del training node más cercano.
        Busca vecindario de orden 4.
        '''

        # Tomar el grafo no dirigido
        undirected_graph = self.graph.copy().to_undirected()

        # Iterar sobre los nodos que no están en el conjunto de entrenamiento
        for v in self.set_no_train_nodes:

            # Buscar un nodo de entrenamiento cercano
            train_encontrado = False
            label_elegir = None

            # hacer una lista de niveles i.e vecinos de varios ordendes
            niveles = [[v]]  # inicializar con el nodo v
            for orden in range(4): # vecindario de orden 4, a lo mas
                # calcular el siguiente nivel
                # estos son los vecinos de los nodos actuales
                siguiente_nivel = []
                # por cada nodo en el nivel actual
                for nodo in niveles[orden]:
                    # por cada vecino
                    for vecino in undirected_graph.neighbors(nodo):
                        # verificar si el vecino es un nodo de entrenamiento
                        if vecino in self.set_train_nodes:
                            # tomar su label, ya dejar de buscar
                            label_elegir = self.graph.nodes[vecino][self.name_label_y]
                            train_encontrado = True
                            break
                        # añadirlo para construir el siguiente nivel
                        siguiente_nivel.append(vecino)
                    # si ya se encontro train en el nivel
                    if train_encontrado:
                        break
                # salir de todos los loops si se encuentra train
                if train_encontrado:
                    break
                # poner todo el nivel
                niveles.append(siguiente_nivel)  # Agregar el nuevo nivel

            # asignar un label aleatorio si no se encontró un nodo de entrenamiento
            if not train_encontrado:
                label_elegir = np.random.randint(self.K)

            # ssignar como predicción de mle y de map
            self.graph.nodes[v]['prediccion_mle_0'] = label_elegir
            self.graph.nodes[v]['prediccion_map_0'] = label_elegir

    # -------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Funciones auxiliares


    # obtener p o s
    def get_vector_nodos_subject(self, labels):
        '''
        Dados unos labels
        Ponerlo en formato de vector
        Esto se usa para construir los vectores p y s
        '''

        # iniciar vacio
        vector_creado = np.zeros(self.K, int)

        # por cada label
        for label in labels:
            # poner uno en el vector
            vector_creado[label] += 1
        return vector_creado


    # tomar el label de un nodo
    def tomar_label_nodo(self, nodo, iteracion_prediccion, metodo):
        '''
        Se pasa un nodo
        un indice de iteracion de prediccion
        y el metodo (mle o map)

        Se devuelve el label del nodo para esa iteracion de prediccion

        Si el nodo es de train, se devuelve su label real (si ignora el indice y el metodo)
        Si el nodo no es de train, se devuelve la prediccion del label
        en la iteracion deseada con el metodo especificado
        '''

        # si es de train
        if nodo in self.train_nodes:
            # subject real
            return self.graph.nodes[nodo][self.name_label_y]

        # no es de train
        else:
            # devolver una prediccion
            # usar la iteracion y el metodo
            return self.graph.nodes[nodo][f"prediccion_{metodo}_{int(iteracion_prediccion)}"]



    # obtener vector s
    def get_vector_s(self, nodo, iteracion, metodo):
        '''
        Obtiene el vector de s descendencia de un nodo
        en una cierta itearcion, con un metodo dado
        Usando labels de la iteracion pasada
        '''

        # tomar los labels de la descendenica
        labels_interes = np.array([ self.tomar_label_nodo(nodo_descendencia,
                                                         iteracion_prediccion = iteracion - 1,
                                                         metodo = metodo)
                                     for nodo_descendencia in  self.graph.successors(nodo)])
        # transformar en vector
        vector_s = self.get_vector_nodos_subject(labels_interes)

        return vector_s


    # obtener vector p
    def get_vector_p(self, nodo, iteracion, metodo):
        '''
        Obtiene el vector de p ascendencia de un nodo
        en una cierta itearcion, con un metodo dado
        Usando labels de la iteracion pasada
        '''

        # tomar los labels de la ascendencia
        labels_interes = np.array([ self.tomar_label_nodo(nodo_ascendencia,
                                                         iteracion_prediccion = iteracion - 1,
                                                         metodo = metodo)
                                     for nodo_ascendencia in  self.graph.predecessors(nodo)])
        # transformar en vector
        vector_p = self.get_vector_nodos_subject(labels_interes)

        return vector_p

    # ----------------------------------------------------------------

    # funcion de log multinomial
    def log_multinomial_pmf(self, vector_z, parametros_P, parametro_d):
        '''
        Dado:
        un vector z
        una lista de los parametros p (entonces una matriz)
        y un parametro d

        Calcular la log pmf de una distribucion multinomial
        con parametros d y p, para cada fila p de P
        evaluar en el vector z
        '''

        # crear las distribuciones multinomiales para cada fila de la matriz
        # todas con el mismo parametro d
        distribuciones = multinomial(parametro_d, parametros_P)

        # para cada una de estas distribuciones, calcular la log pmf de z
        log_probs = np.log(distribuciones.pmf(vector_z))

        return log_probs

    # ----------------------------------------------------------------
    # -------------------------------------------------------------
    # Inferencia de un nodo


    # toma un nodo del grafo que no sea train
    # y una iteracion de la inferencia
    # infiere el subject usando la informacion de iteracion pasada
    def inferir_label_nodo_iteracion(self, nodo, iteracion, metodo):
        '''
        Argumentos:
            nodo        - nodo del grafo que no sea de entrenamiento
            iteracion   - int  -  numero de iteracion en la que se hace inferencia
            metodo      - metodo usado para la inferencia, puede ser mle o map
            parametros  - parametros del modelo

        Se usan las labels producto de la iteracion pasada para nodos no train
        Para nodos en train se usa su label real, pues se conoce.
        Se hace inferencia sobre el label del nodo actual
        Es decir, hace inferencia en la iteracion, dada la inferencia de la iteracion anterior

        Para la inferencia, toma el label que maximize un score, puede ser el likelihood o el posterior probability
        Es decir, se hacer mle o map
        Se minimiza el negative log likelihood (y se ponen priors para map)
        '''

        # tomar info del nodo
        d_out = self.graph.nodes[nodo]['out_degree']
        d_in = self.graph.nodes[nodo]['in_degree']
        vector_s_descendencia = self.get_vector_s(nodo, iteracion, metodo)
        vector_p_ascendencia = self.get_vector_p(nodo, iteracion, metodo)


        # iniciar los scores a minimizar en 0
        # guardar un score por cada subject
        # al final tomar el subject que minimize
        scores_minimizar = np.zeros(self.K)

        # tomar los valores del texto, ya se tienen
        scores_minimizar -= self.graph.nodes[nodo]['log_omega_evaluated']

        # añadir lo del grado de entrada y salida
        scores_minimizar -= self.parameters["matrix_log_psi"][:, min(d_in, self.config["D_in_max"])]
        scores_minimizar -= self.parameters["matrix_log_phi"][:, min(d_out, self.config["D_out_max"])]


        # añadir lo de la ascendencia
        scores_minimizar -= self.log_multinomial_pmf(vector_p_ascendencia,
                                                     parametros_P = self.parameters["Xi"],
                                                     parametro_d = d_in)

        # añadir lo de la descendencia
        scores_minimizar -= self.log_multinomial_pmf(vector_s_descendencia,
                                                     parametros_P = self.parameters["Theta"],
                                                     parametro_d = d_out)

        # añadir las priors (solo MAP)
        if metodo.lower() == "map":
            scores_minimizar -= self.parameters["log_pi"]


        # ya que se tienen los scores
        # ver cual es el subject que las minimiza
        prediccion_y = scores_minimizar.argmin()

        # poner esta prediccion como argumento de la iteracion actual
        self.graph.nodes[nodo][f"prediccion_{metodo}_{int(iteracion)}"] = prediccion_y

    # --------------------------------------------------------------------

    # Funcion de iteracion de inferencia

    # hacer toda una iteracion de predicciones
    def iteracion_inferencia(self, num_iteracion, metodo):
        '''
        Una iteracion de la inferencia con el metodo especificado
        '''

        # por cada nodo no en train
        for v in tqdm(self.set_no_train_nodes):

            # hacer prediccion en ese nodo
            self.inferir_label_nodo_iteracion(nodo = v,
                                              iteracion = num_iteracion,
                                              metodo = metodo)

    # --------------------------------------------------------------------

    # Comparar iteraciones

    # obtener ciertas predicciones
    def obtener_predicciones(self, nodos_interes, idx_iteracion, metodo):
        '''
        Argumentos:
        conjunto de nodos
        indice de una iteracion de inferencia
        metodo para realizar inferencia

        Devuelve las predicciones en el conjunto especificado
        en la iteracion desada con el metodo
        '''

        # poner las predicciones en un diccionario
        predicciones = {v: self.tomar_label_nodo(v, idx_iteracion, metodo)
                        for v in nodos_interes}

        return predicciones


    # ver si las predicciones cambian
    def comparar_inferencia_iteracion_pasada(self, idx_iteracion, metodo):
        '''
        Dado el indice de una iteracion de infenrencia y un metodo
        Ver si las predicciones de esa iteracion cambian mucho con respecto
        de la iteracion pasada con el mismo metodo
        Usar el conjunto de validacion para checar esto
        '''

        # tomar predicciones de la iteracion actual
        predicciones_actual_dict = self.obtener_predicciones(self.val_nodes, idx_iteracion, metodo)
        # tomar las predicciones pasadas
        predicciones_pasadas_dict = self.obtener_predicciones(self.val_nodes, idx_iteracion - 1, metodo)

        # pasar a arrays
        predicciones_actual = np.array([predicciones_actual_dict[v] for v in self.val_nodes])
        predicciones_pasada = np.array([predicciones_pasadas_dict[v] for v in self.val_nodes])

        # ver cuantos son iguales
        num_nodos_iguales = (predicciones_actual == predicciones_pasada).sum()
        # porcentaje
        porcentaje_iguales = num_nodos_iguales/len(self.val_nodes)

        print("\nIn validation nodes")
        print(f"Inference on interation {idx_iteracion} is {round(100*porcentaje_iguales)}% equal to the last iteration")


    # compara mle y map
    def comparar_metodos_inferencia(self, idx_iteracion):
        '''
        Usando los nodos de validacion

        Comapra las predicciones en una iteracion con ambos metodos
        Ve que tan similares son
        '''

        # obtener predicciones mle
        predicciones_mle_dict = self.obtener_predicciones(self.val_nodes, idx_iteracion, metodo = "mle")
        # obtener predicciones map
        predicciones_map_dict = self.obtener_predicciones(self.val_nodes, idx_iteracion, metodo = "map")

        # pasar a array
        predicciones_mle = np.array([predicciones_mle_dict[v] for v in self.val_nodes])
        predicciones_map = np.array([predicciones_map_dict[v] for v in self.val_nodes])

        # ver cuantos son iguales
        num_nodos_iguales = (predicciones_mle == predicciones_map).sum()
        # porcentaje
        porcentaje_iguales = num_nodos_iguales/len(self.val_nodes)

        print("\nIn validation nodes")
        print(f"Inference on iteration {idx_iteracion} yield ML and MAP predictions {round(100*porcentaje_iguales)}% equal")


    # -------------------------------------------------------------------------
    # Funcion principal para inferencia

    def hacer_inferencia(self):

        '''
        Hacer inferencia con el modelo probabilistico
        Se usan los metodos de MLE y MAP
        '''

        print("Precompute omega function")
        self.precomputar_log_funcion_omega()
        self.precomputar_grados() # y de paso los grados

        # hacer la iteracion 0
        print("Inference iteration 0")
        self.hacer_predicciones_iteracion_0()

        # evaluar
        print(self.evaluar_iteracion_metodo(0, "mle"))

        # tantas iteraciones como se quiera
        for it in range(1, self.config["num_iterations"] + 1):

            # indicar
            print("-"*300)
            print(f"Iteration {it}")

            # hacer inferencia usando mle
            print("\nInference using ML")
            self.iteracion_inferencia(it, "mle")
            # evaluar
            print(self.evaluar_iteracion_metodo(it, "mle"))
            # compara con la prediccion mle anterior
            self.comparar_inferencia_iteracion_pasada(it, "mle")

            # hacer inferencia usando map
            print("\nInference using MAP")
            self.iteracion_inferencia(it, "map")
            # evaluar
            print(self.evaluar_iteracion_metodo(it, "map"))
            # comparar con la prediccion map anterior
            self.comparar_inferencia_iteracion_pasada(it, "map")

            print("")
            # comparar mle y map entre si
            self.comparar_metodos_inferencia(it)

        # end iteraciones
    # end funcion inferencia

    # --------------------------------------------------------
    # Evaluar las estadisticas en las iteraciones

    # hacer un diccionario con estadisticas
    def calcular_estadisticas(self, y_true, y_pred):
        '''
        Funcion auxuliar para calcular estadisticas
        '''

        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        return {"f1_macro" :f1_macro, "f1_weighted" :f1_weighted, "f1_micro" :f1_micro}


    # evaluar una iteracion
    def evaluar_iteracion_metodo(self, idx_iteracion, metodo):
        '''
        Se especifica el numero de una iteracion y el metodo

        Se evaluan las predicciones correspondientes en nodos de validacion
        Se devuelven las metricas de estas predicciones
        '''

        # delimitar labels reales
        labels_reales = np.array([self.graph.nodes[v][self.name_label_y]
                                  for v in self.val_nodes])

        # tomar las prediccioens de la iteracion y el metodo
        predicciones = self.obtener_predicciones(self.val_nodes,
                                                 idx_iteracion, metodo)

        # pasar a array
        predicciones = np.array([predicciones[v] for v in self.val_nodes])

        # calcular estadisticas
        estadisticas_iteracion = self.calcular_estadisticas(y_true = labels_reales,
                                                            y_pred = predicciones)

        return estadisticas_iteracion

    # ------------------------------------------------------------------
    # Funciones para graficar


    # funcion auxiliar para ver metricas
    def plot_metrica_mle_map(self, estadisticas_iteraciones_mle,
                             estadisticas_iteraciones_map, metrica, ax):
        '''
        Dadas las estadisticas de MLE y MAP y una metrica

        Graficar ambos metodos a traves de las iteraciones
        '''

        # dibujar la metrica de mle a traves de las iteraciones
        ax.plot([dict_stats[metrica] for dict_stats in estadisticas_iteraciones_mle], label = "MLE")

        # dibujar map
        ax.plot([dict_stats[metrica] for dict_stats in estadisticas_iteraciones_map], label = "MAP")

        ax.legend()
        ax.set_title(metrica)


    def plot_metrics(self, estadisticas_iteraciones_mle,
                     estadisticas_iteraciones_map):
        '''
        Grafica las metricas para MLE y para MAP durante inferencia
        '''

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        # graficar las 3 metricas
        self.plot_metrica_mle_map(estadisticas_iteraciones_mle, estadisticas_iteraciones_map,  "f1_macro", ax[0])
        self.plot_metrica_mle_map(estadisticas_iteraciones_mle, estadisticas_iteraciones_map,  "f1_weighted", ax[1])
        self.plot_metrica_mle_map(estadisticas_iteraciones_mle, estadisticas_iteraciones_map,  "f1_micro", ax[2])

        plt.show()


    # para estadisticas de las iteraciones
    def analizar_estadisticas_iteraciones_val(self, numero_iteraciones):
        '''
        Para el numero de iteraciones que se especifique

        Visualizar graficas de calidad de predicciones de MLE y MAP
        Usando datos de validacion

        Devuelve la iteracion que proporciona las mejores proporciones
        para cada metodo
        '''

        print(f"\nAnalize statistics of results in {len(self.val_nodes)} validation nodes")

        # guardar estadisticas de cada iteracion para cada metodo
        estadisticas_iteraciones_mle =  []
        estadisticas_iteraciones_map =  []

        # por cada iteracion
        for idx_iter in range(numero_iteraciones+1):

            # agregar metricas de esta iteracion

            # agregar para mle
            estadisticas_iteraciones_mle.append(self.evaluar_iteracion_metodo(idx_iter, "mle"))
            # agregar para map
            estadisticas_iteraciones_map.append(self.evaluar_iteracion_metodo(idx_iter, "map"))

        # visualizar los datos
        self.plot_metrics(estadisticas_iteraciones_mle, estadisticas_iteraciones_map)


        # Ver el mejor desempeño para cada metodo

        # delimitar la metrica importante
        metrica_principal = self.config["Metric"]

        # tomar esta metrica solo de mle y de map
        metrica_mle = np.array([dict_stats[metrica_principal]
                                for dict_stats in estadisticas_iteraciones_mle])
        metrica_map = np.array([dict_stats[metrica_principal]
                                for dict_stats in estadisticas_iteraciones_map])

        # para cada metodo, tomar la mejor iteracion y el mejor valor alcanzado
        # para mle
        best_iter_mle = metrica_mle.argmax()
        best_metric_mle = metrica_mle.max()
        # para map
        best_iter_map = metrica_map.argmax()
        best_metric_map = metrica_map.max()

        # poner la mejor iteracion para cada metodo
        # y tambien la mejor metrica para cada metodo
        dict_best_iter = {"mle": best_iter_mle, "map": best_iter_map}
        dict_best_metric = {"mle": best_metric_mle, "map": best_metric_map}

        return dict_best_iter, dict_best_metric


    # ----------------------------------------------------------
    # Actualizar hyperparametros y salvar predicciones


    def save_prediction(self, iter_mle, iter_map, prediction_path):

        '''
        Toma dos iteraciones de inferencia, una para MLE otra para MAP

        Con estas iteraciones, guarda las predicciones con ambos metodos
        Usando todos los nodos

        Para las predicciones se usa el nombre de los labels en: decode_label
        '''

        # tomar todos los nodos, para hcer prediccion ahi
        nodos_grafo = set(list(self.graph.nodes()))

        #  predecir en todos los nodos

        # para mle
        predicciones_finales_mle = self.obtener_predicciones(nodos_interes = nodos_grafo,
                                                             idx_iteracion = iter_mle,
                                                             metodo = "mle")
        # para map
        predicciones_finales_map = self.obtener_predicciones(nodos_interes = nodos_grafo,
                                                             idx_iteracion = iter_map,
                                                             metodo = "map")



        # hacer que las predicciones tengan los nombres de las labels
        # para mle
        predicciones_finales_mle = {v: self.decode_label[label]
                                    for v, label in predicciones_finales_mle.items()}
        # para map
        predicciones_finales_map = {v: self.decode_label[label]
                                    for v, label in predicciones_finales_map.items()}

        # salvar las predicciones finales

        # salvar mle
        with open(prediction_path + 'probabilistic_graph_model_ML.pkl', 'wb') as f:
            pickle.dump(predicciones_finales_mle, f)
        # salvar map
        with open(prediction_path + 'probabilistic_graph_model_MAP.pkl', 'wb') as f:
            pickle.dump(predicciones_finales_map, f)


    def update_hyperparameters(self, mle_metric, map_metric,
                               hyperparameters_file):
        '''
        Al finalizar la inferencia
        Guardar las metricas obtenidas para los hyperparametros usados
        '''

        # hacer strings los valores de los parametros
        config_salvar = {key: str(value) for key, value in self.config.items()}

        # agregar metricas alcanzadas
        config_salvar["MLE_metric"] = mle_metric
        config_salvar["MAP_metric"] = map_metric


        # hacer un df con esta info
        new_df = pd.DataFrame([config_salvar])

        # intenta cargar un archivo existente
        try:

            # leer
            df = pd.read_csv(hyperparameters_file, na_values=None, keep_default_na=False)
            # agregar nueva fila, con la info de esta inferencia
            df = pd.concat([df, new_df], ignore_index=True)
            # quitar repetidos
            df = df.drop_duplicates(keep='last')
            # ordenar segun metrica
            df = df.sort_values(by=['MLE_metric', "MAP_metric"], ascending = False)
            # guardar el archivo modificado
            df.to_csv(hyperparameters_file, index=False)

        # si no se puede abrir, no existe
        except FileNotFoundError:

            # guardar solo la info de esta iteracion
            new_df.to_csv(hyperparameters_file, index=False)

    # ---------------------------------------------------------------
    # --------------------------------------------------------------
    # Funcion principal

    def probabilistic_inference_complete(self, config,
                                         folder_parameters = "./",
                                         prediction_path = "./",
                                         hyperparameters_file = "./"):
        '''
        Hace todo el proceso de inferencia probabilistica con el modelo
        config - diccionario de hyperparametros
        folder_parameters - folder donde se guardan los parametros
        prediction_path - folder donde guardar las predicciones
        hyperparameters_file - name and path to the hyperparameter file
        '''

        # Estimar parametros
        print("Estimating parameters...")
        self.estimate_parameters(config, folder_parameters)
        print("Done!\n")

        # poner config como atributo
        self.config = config

        # Inferencia
        self.hacer_inferencia()

        # ver y analizar estadisticas, tomar mejor iteracion para cada metodo
        dict_best_iter, dict_best_metric = self.analizar_estadisticas_iteraciones_val(numero_iteraciones =
                                                                                      config["num_iterations"])

        # tomar la mejor iteracion de cada metodo
        best_iter_mle = dict_best_iter['mle']
        best_iter_map = dict_best_iter['map']

        # tomar las mejore metricas
        best_metric_mle = dict_best_metric['mle']
        best_metric_map = dict_best_metric['map']

        # poner como atributos
        self.best_iter_mle_ = best_iter_mle
        self.best_iter_map_ = best_iter_map
        self.best_metric_mle_ = best_metric_mle
        self.best_metric_map_ = best_metric_map

        # indicar
        print(f"MLE achieves the best {config['Metric']} {round(best_metric_mle, 5)} in iteration {best_iter_mle}")
        print(f"MAP achieves the best {config['Metric']} {round(best_metric_map, 5)} in iteracion {best_iter_map}")

        # ver desempeño mejor iteracion mle
        print("Mejor desempeño MLE")
        print(self.evaluar_iteracion_metodo(best_iter_mle, 'mle'))
        # ver desempeño mejor iteracion map
        print("Mejor desempeño MAP")
        print(self.evaluar_iteracion_metodo(best_iter_map, 'map'))

        # guardar los resultados de estos hyperparametros
        self.save_prediction(best_iter_mle, best_iter_map,
                             prediction_path = prediction_path)

        # actualizar hyperparametros
        self.update_hyperparameters(best_metric_mle, best_metric_map,
                                    hyperparameters_file = hyperparameters_file)
    # end probabilistic_inference_complete

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # Example of prediction

    def see_example_prediction(self, node_v, iteration, method, num_top_labels = 3):
        '''
        See the proces behind the prediction of a single node
        Interpretability of predictions

        Given a node v, an iteration of interest and a method (mle, map)
        explain the prediction of label if that node, on that iteration with that method

        Show the best num_top_labels labels for this node
        '''

        print(f"Prediction of node {node_v} in iteration {iteration} using {method}\n")
        division_v = self.graph.nodes[node_v][self.name_division]
        print(f"{division_v} node")
        if  division_v != "useless":
            true_label = self.graph.nodes[node_v][self.name_label_y]
            print(f"True label: {true_label} ({self.decode_label[true_label]})")
        print("")
        print("-"*50)

        # info of the node
        x_v = self.graph.nodes[node_v][self.name_atributes_x]
        d_in = self.graph.nodes[node_v]['in_degree']
        predecessors_v = [v for v in self.graph.predecessors(node_v)]
        d_out = self.graph.nodes[node_v]['out_degree']
        succcesors_v = [v for v in self.graph.successors(node_v)]

        # informar
        print(f"Information of node {node_v}:")
        print(f"Text atributes: {x_v}")
        print(f"\nIn Degree: {d_in}")
        if d_in > 0:
            print(f"Predecesors: {predecessors_v}")
            # print every predecesor
            for u in predecessors_v:
                # si es train se sabe el label
                if u in self.train_nodes:
                    print(f"\t{u} - training node. Known label: {self.graph.nodes[u][self.name_label_y]}")
                # no traini node, no se sabe el label
                else:
                    print(f"\t{u} - not training node. Unknown label")
        print(f"Out Degree: {d_out}")
        if d_out > 0:
            print(f"Successors: {succcesors_v}")
            # print every successor
            for u in succcesors_v:
                # si es train se sabe el label
                if u in self.train_nodes:
                    print(f"\t{u} - training node. Known label: {self.graph.nodes[u][self.name_label_y]}")
                # no traini node, no se sabe el label
                else:
                    print(f"\t{u} - not training node. Unknown label")



        # la iteracion ayuda a nodos para los que no se sabe la label
        print("-"*50)
        print(f"\nPredict label of {node_v} in iteration {iteration}")
        print(f"Use {method} predictions on iteration {iteration - 1} for the neihboors of {node_v}\n")


        # volver a iterar en predecesors y successors, con labels de iteracion
        if d_in > 0:
            print(f"Predecesors: {predecessors_v}")
            # print every predecesor
            for u in predecessors_v:
                # si es train se sabe el label
                if u in self.train_nodes:
                    print(f"\t{u} - training node. Known label: {self.graph.nodes[u][self.name_label_y]}")
                # no training node, poner la prediccion pasada
                else:
                    print(f"\t{u} - not training node. {method} prediction on iteration {iteration - 1}: {self.tomar_label_nodo(u, iteracion_prediccion = iteration - 1, metodo = method)}")
        if d_out > 0:
            print(f"Successors: {succcesors_v}")
            # print every predecesor
            for u in succcesors_v:
                # si es train se sabe el label
                if u in self.train_nodes:
                    print(f"\t{u} - training node. Known label: {self.graph.nodes[u][self.name_label_y]}")
                # no training node, poner la prediccion pasada
                else:
                    print(f"\t{u} - not training node. {method} prediction on iteration {iteration - 1}: {self.tomar_label_nodo(u, iteracion_prediccion = iteration - 1, metodo = method)}")


        # tomar los vectores
        vector_p_ascendencia = self.get_vector_p(node_v, iteration, method)
        vector_s_descendencia = self.get_vector_s(node_v, iteration, method)
        # informar
        print("\nVector p_v:")
        print(vector_p_ascendencia)
        print("Vector s_v:")
        print(vector_s_descendencia)

        # ya se tiene la info para hacer la inferencia
        print("")
        print("-"*50)
        #print("With this information we can make inference")


        # iniciar los scores a minimizar en 0
        # guardar un score por cada subject
        scores_minimizar = np.zeros(self.K)

        # tomar los valores del texto, ya se tienen
        atribute_discr = -self.graph.nodes[node_v]['log_omega_evaluated']
        #print("\nAtribute discrepancy:")
        #print(atribute_discr)
        scores_minimizar += atribute_discr

        # Predecessor Count Discrepancy
        pred_count_discr = -self.parameters["matrix_log_psi"][:, min(d_in, self.config["D_in_max"])]
        #print("\nPredecessor Count Discrepancy:")
        #print(pred_count_discr)
        scores_minimizar += pred_count_discr

        # Successor Count Discrepancy
        succ_count_discr =  -self.parameters["matrix_log_phi"][:, min(d_out, self.config["D_out_max"])]
        #print("\nSuccessor Count Discrepancy:")
        #print(succ_count_discr)
        scores_minimizar += succ_count_discr

        # label Predecessors Discrepancy
        pred_label_discr = -self.log_multinomial_pmf(vector_p_ascendencia,
                                                     parametros_P = self.parameters["Xi"],
                                                     parametro_d = d_in)
        #print("\nLabel Predecessors Discrepancy:")
        #print(pred_label_discr)
        scores_minimizar += pred_label_discr


        # Label Successors Discrepancy
        succ_label_discr = -self.log_multinomial_pmf(vector_s_descendencia,
                                                     parametros_P = self.parameters["Theta"],
                                                     parametro_d = d_out)
        #print("\nLabel Successors Discrepancy:")
        #print(succ_label_discr)
        scores_minimizar += succ_label_discr


        # añadir las priors (solo MAP)
        if method.lower() == "map":
            prior_discr = -self.parameters["log_pi"]
            #print("\nPrior Discrepancy:")
            #print(prior_discr)
            scores_minimizar += prior_discr


        # sumar todas
        #print("\nAll discrepancies")
        #print(scores_minimizar)
        #print("-"*50)

        # ver los labels con menor discrepancy
        top_labels = np.argsort(scores_minimizar)[:num_top_labels]

        print(f"\nTop {num_top_labels} labels with lowest discrepancy")
        print(top_labels)

        # por cada una de las buenas labels
        for label_buena in top_labels:
            print(f"\nLabel: {label_buena} ({self.decode_label[label_buena]})")
            print(f"\tAtribute discrepancy: {atribute_discr[label_buena]}")
            print(f"\tPredecessor Count Discrepancy: {pred_count_discr[label_buena]}")
            print(f"\tSuccessor Count Discrepancy: {succ_count_discr[label_buena]}")
            print(f"\tLabel Predecessors Discrepancy: {pred_label_discr[label_buena]}")
            print(f"\tLabel Successors Discrepancy: {succ_label_discr[label_buena]}")
            if method.lower() == "map":
                print(f"\tPrior Discrepancy: {prior_discr[label_buena]}")
            print(f"\tTotal discrepancy: {scores_minimizar[label_buena]}")

        print("-"*30)

        # prediccion final
        final_prediction = self.graph.nodes[node_v][f"prediccion_{method}_{int(iteration)}"]
        print(f"\nFinal prediction: {final_prediction} ({self.decode_label[final_prediction]})")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # Evaluate predictions on test data


    def evaluate_test_nodes(self, iteration, method):
        '''
        Given an iteration and a method
        Evaluate the corresponding predictions on the test data
        '''

        # delimitar labels reales
        labels_reales_test = np.array([self.graph.nodes[v][self.name_label_y]
                                       for v in self.test_nodes])

        # tomar las prediccioens de la iteracion y el metodo
        predicciones_test = self.obtener_predicciones(self.test_nodes,
                                                      iteration, method)
        # pasar a array
        predicciones_test = np.array([predicciones_test[v] for v in self.test_nodes])

        # calcular estadisticas
        estadisticas_iteracion = self.calcular_estadisticas(y_true = labels_reales_test,
                                                            y_pred = predicciones_test)

        return estadisticas_iteracion

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
