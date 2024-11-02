# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:52:04 2024

@author: diego
"""



from collections import Counter
from scipy.stats import chisquare
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


'''
Helper functions to model the functions phi and psi
'''


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


def frecuency_probabilities(sample, smoothing_parameter = 0.01, max_value = 10000):
    '''
    Given a sample of non negative integers in [0, max_value]
    Returns frecuency probabilities using aditive smoothing
    '''
    
    # iniciar, con zeros, longuitud maxima 
    proporciones = np.zeros(max_value + 1)
    
    # Contar las ocurrencias de cada valor en la muestra
    for valor in sample:
        if 0 <= valor <= max_value:  # Asegurarse de que el valor esté en el rango
            proporciones[valor] += 1
    
    # sumar un suavizado 
    proporciones += smoothing_parameter
    # Convertir las cuentas en proporciones 
    proporciones = proporciones / sum(proporciones)
    
    return proporciones


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


# comprueba si el vector es una funcion de probabilidad
def comprobar_vector_probabilidad(vector):

    # comprobar que todos los elementos sean no negativos
    if any(elemento < 0 for elemento in vector):
        return False
    # comprobar que la suma de todos los elementos sea aproximadamente 1
    if not np.isclose(sum(vector), 1):
        return False

    # es una función de probabilidad
    return True


# obtener k celdas equiprobables dado un vector de probabilidad
# se devuelven los indices de las celdas
def obtener_k_celdas_equiprobables(vector, k):
    '''
    Dado un vector de probabilidad,
    crear k celdas tal que tengan la misma probabilidad
    Cada celda es un intervalo de incices [a, b)
    '''

    # ir contando cuantas celdas van
    num_celdas = 0

    # ver cuanta proba hay en cada celeda
    probabilidad_celda = 1/k

    # hacer una lista de k elementos
    # es el inicio y el fin + 1 de los elementos de cada celda

    # inciar vacio
    celdas = []

    # in moviendose en el vector
    inicio_celda = 0

    for idx in range(len(vector)):

        # ver lo que va de la celda, con las celdas anteriores
        proba_celda_y_anterior = sum(vector[:idx])

        # si ya se pasa de lo que debe de tener
        if proba_celda_y_anterior >= probabilidad_celda * (1 + num_celdas):

            # si es que ya se sumo a 1
            if proba_celda_y_anterior == 1:
                # no terminar aca, terminar hasta el final
                celdas.append([inicio_celda, len(vector)])

                # ver que sean k
                assert k == len(celdas)

                # ya terminar
                return celdas

            # aun no suma a 1

            # si es que solo falta una celda
            if len(celdas) == k-1:
                # esta celda va a hasta el final
                celdas.append([inicio_celda, len(vector)])
                # ver que sean k
                assert k == len(celdas)

                # ya terminar
                return celdas

            # aun no suma a 1
            # y se tiene que poner al menos otra celda

            # terminar esta celda
            # meterla a las listas
            celdas.append([inicio_celda, idx])

            # actualizar
            inicio_celda = idx
            num_celdas += 1

    # contruir la utlima celda, si es que no es el final de una
    if inicio_celda != len(vector):
        celdas.append([inicio_celda, len(vector)])

    # ver que haya funcionado
    assert k == len(celdas)

    # devolver
    return celdas


# sumar las etradas de un vector segun
# los indices de unas celdas
def sumar_segun_celdas(vector, celdas):
    '''
    Toma un vector, y unas celdas (indices)
    devuelve la suma del vector en cada celda
    '''

    # ir guardadno la suma de cada celda
    vector_sumado = []

    # ir iterando en cada celda
    for celda in celdas:

        # sumar y agregar
        vector_sumado.append(sum(vector[celda[0]:celda[1]]))

    return vector_sumado


# obtenre las ocurrencias 
def get_ocurrencias_dominio(muestra, dominio):
    '''
    Dado una muestra, y su dominio,
    devovler las ocurrencias de cada elemento del dominio
    devolviendo un vector con el mismo tamaño que el dominio
    '''

    # ocurrecias
    ocurrencias = np.array([sum([1 for x in muestra if x == elemento_dominio])
                  for elemento_dominio in dominio])

    # ver que se tengan todas las ocurrencias
    if sum(ocurrencias) == len(muestra):
        return ocurrencias
    
    # si no es el caso, hay valroes en la muestra fuera del dominio
    # ver cuales son
    outliers = [x for x in muestra if x not in dominio]
    
    # contar excepeciones
    outliers_pequeños = 0
    outliers_grandes = 0
    
    # por cada outlier
    for x in outliers:
        # si es menor
        if x < min(dominio):
            outliers_pequeños = outliers_pequeños + 1
        elif x > max(dominio):
            outliers_grandes = outliers_grandes + 1
            
    # decir
    if outliers_pequeños > 0:
        print(f"WARNING: {outliers_pequeños} outliers pequeños")
    if outliers_grandes > 0:
        print(f"WARNING: {outliers_grandes} outliers grandes")
        
    # arreglar, poner en los extremos
    ocurrencias[0] = ocurrencias[0] + outliers_pequeños
    ocurrencias[-1] = ocurrencias[-1] + outliers_grandes
    
    # comprobar que ahora si
    assert sum(ocurrencias) == len(muestra)
    
    return ocurrencias



# ver si los valores esperados en celdas son validos
def valid_expected(valores_esperados):
    
    # si hay uno menor a 1 no es valido
    if any([e<1 for e in valores_esperados]):
        return False
    
    # ver cuantos son menos que 5
    menores_cinco = len([e for e in valores_esperados if e < 5])
    porcentaje_menor_a_cinco = menores_cinco/len(valores_esperados)
    
    # si mas del 20% son menores a 5 no son validos
    if porcentaje_menor_a_cinco > 0.2:
        return False 
    
    # si no pasa nada de esto
    return True



# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Zero truncated power law



# pmf sin nromalizar
def functional_zero_truncated_pw(d, kappa, lamda):
    '''
    Funcional de la distribucion propuesta
    zero truncated power law
    
    Es decir, la pmf no normalizada para valores positivos
    '''
    
    # devolver para cuando d>0
    return np.power(d, -kappa) * np.exp(-d*lamda)


# log pmf sin nromalizar
def log_functional_zero_truncated_pw(d, kappa, lamda):
    '''
    Logaritmo del funcional de la distribucion propuesta
    zero truncated power law
    
    Es decir, la pmf no normalizada para valores positivos, en nogaritmo
    '''
    
    # devolver para cuando d>0
    return -kappa*np.log(d) -d*lamda


# pmf 
def pmf_zero_truncated_pw(d, beta, kappa, lamda,
                          C = None, D_max = 1000):
    '''
    Funcion de probabilidad de la distribucion propuesta
    zero truncated power law
    Parametros: beta, kappa, lambda
    
    C es la constante normalizadora 
    (funcional evaluado en valores mayores a 0)
    
    D_max es el valor maximo considerado para d
    '''

    # poner las log probabilidades en un vector (si d es vector)
    resultado = np.zeros_like(d, float)
    
    # si no se tiene la constante normalizadora, calcularla
    if C is None:
        C = 1 / (functional_zero_truncated_pw(np.arange(1, D_max), kappa, lamda)).sum()
        
    # donde es 0 poner beta
    resultado[d==0] = beta
    
    # donde d>0, poner el funcional normalizado
    resultado[d!=0] = (1 - beta) * C * functional_zero_truncated_pw(d[d!=0], kappa, lamda)
    
    return resultado



# log pmf 
def log_pmf_zero_truncated_pw(d, beta, kappa, lamda,
                              C = None, D_max = 1000):
    '''
    Logaritmo de la funcion de probabilidad de la distribucion propuesta
    zero truncated power law
    Parametros: beta, kappa, lambda
    
    C es la constante normalizadora 
    (funcional evaluado en valores mayores a 0)
    
    D_max es el valor maximo considerado para d
    '''

    # poner las log probabilidades en un vector (si d es vector)
    resultado = np.zeros_like(d, float)
    
    # si no se tiene la constante normalizadora, calcularla
    if C is None:
        C = 1 / (functional_zero_truncated_pw(np.arange(1, D_max), kappa, lamda)).sum()
        
    # donde es 0 poner log beta
    resultado[d==0] = np.log(beta)
    
    # donde d>0, poner el funcional normalizado, todo en logaritmo
    resultado[d!=0] = np.log(1 - beta) + np.log(C) +  log_functional_zero_truncated_pw(d[d!=0], kappa, lamda)
    
    return resultado




# estimar parametros
def estimate_zero_truncated_pw(sample, D_max = 1000):
    '''
    Given a sample, estimate the parameters of the proposed distribution
    zero truncated power law
    
    Estimate: beta, kappa, lambda
    '''
    sample = np.array(sample)
    
    # primero, beta se estima como el porcentaje de datos que son cero
    estimacion_beta = (sample == 0).sum()/len(sample)
    
    # tomar valores positivos de la muestra para estimar lo que sigue
    positive_sample = sample[sample>0]
    
    # minimizar el minus log likelihood para kappa y lambda
    
    # definir el minus_log_likelihood, como funcion de los parametros
    def compute_minus_log_likelihood(kappa_lamda):
    
        # separar parametros
        kappa = kappa_lamda[0]
        lamda = kappa_lamda[1]
    
        # calcular la constante de estos parametros
        C = 1 / (functional_zero_truncated_pw(np.arange(1, D_max), kappa, lamda)).sum()
        
        # tomar la log probabilidad de cada elemento en la muestra
        log_proba = log_pmf_zero_truncated_pw(d = positive_sample,
                                              beta = estimacion_beta,
                                              kappa = kappa,
                                              lamda = lamda, 
                                              C = C, D_max = D_max)
        
        # devolver minus log likelihood
        return -log_proba.sum()
    
    
    # para estimar los parametros, poner un punto inicial
    punto_inicial = np.array([1, 0.5])
    
    # limites de kappa y lamda
    bounds = [(1e-6, None), (1e-6, None)]
    
    # estimar los parametros como problema de minimizacion
    resultado = minimize(compute_minus_log_likelihood,
                         punto_inicial,
                         method ='nelder-mead',
                         bounds = bounds)
    
    # Obtener los resultados
    parametros_optimizados = resultado.x
    
    # separar los parametros
    estimacion_kappa = parametros_optimizados[0]
    estimacion_lamda = parametros_optimizados[1]
    
    return {'beta': estimacion_beta,
           'kappa': estimacion_kappa,
           'lamda': estimacion_lamda}



# test for zero truncated power law
def analize_zero_truncated_pw(sample, D_max = 1000,
                              k_chi2 = 10, ver = True):
    '''
    Goodness of fit test for the proposed distriution
    Chi square goodness of fit test with k_chi2 cells
    zero truncated power law
    '''
    
    # ver el numero de datos
    n = len(sample)
    if ver:
        print(f"Sample of size {n}")

    # ----------------------------------------------------------

    # obtener laas estimaciones MLE de los parametros
    parametros_optimizados = estimate_zero_truncated_pw(sample)

    # separar
    estimacion_beta = parametros_optimizados['beta']
    estimacion_kappa = parametros_optimizados['kappa']
    estimacion_lamda = parametros_optimizados['lamda']

    # calcular la constante de normalizacion para estos parametros
    C = 1 /(functional_zero_truncated_pw(np.arange(1, D_max), estimacion_kappa, estimacion_lamda)).sum()

    # show parameters
    if ver:
        print("\nEstimated parameters:")
        print(f"beta = {estimacion_beta}")
        print(f"kappa = {estimacion_kappa}")
        print(f"lamda = {estimacion_lamda}")
        print(f"Normalization C = {C}")


    # con estos parametros, delimitar
    # la funcion de probabilidad de H0
    def pmf_estimada(x):
        return pmf_zero_truncated_pw(x,
                                     beta = estimacion_beta,
                                     kappa = estimacion_kappa,
                                     lamda = estimacion_lamda,
                                     C = C, D_max = D_max)
        
    # comprobar que sea funcion de probabilidad en el dominio
    assert np.isclose(sum(pmf_estimada(np.arange(D_max))), 1)


    # ----------------------------------------------------------

    # el dominio lo delimita D_max
    dominio = np.arange(0, D_max)

    # obtener las probabilidades del dominio
    proba_h0_dominio = pmf_estimada(dominio)

    # ----------------------------------------------------------

    # tomar las celdas equiprobables
    celdas = obtener_k_celdas_equiprobables(proba_h0_dominio, k_chi2)
    if ver:
        print("\nCells:")
        print(celdas)
        print("")

    # ----------------------------------------------------------

    # tomar las probabilidades de la distribucion de H0 en las celdas (casi equiprobable)
    proba_h0_celdas = sumar_segun_celdas(proba_h0_dominio, celdas)
    # corregir en caso de haber errores
    proba_h0_celdas[-1] = 1 - sum(proba_h0_celdas[:-1])

    # ver que si sea probabilidad en las celdas
    assert comprobar_vector_probabilidad(proba_h0_celdas)

    # obtener lo esperado en cada celda
    # son solo las probabilidades por el numero de datos
    esperado_celdas = np.array(proba_h0_celdas) * n

    if ver:
        print("Expecteed values in cells")
        print(esperado_celdas)
        
        
    # ver que sean validas las celdas
    while not valid_expected(esperado_celdas):
        
        # bajar k en uno
        k_chi2 = k_chi2-1
        if ver:
            print("\nExpected values not valid")
            print(f"Reduce k to {k_chi2}")
        
        # volver a calcular
        celdas = obtener_k_celdas_equiprobables(proba_h0_dominio, k_chi2)
        
        # esperados
        proba_h0_celdas = sumar_segun_celdas(proba_h0_dominio, celdas)
        proba_h0_celdas[-1] = 1 - sum(proba_h0_celdas[:-1])
        assert comprobar_vector_probabilidad(proba_h0_celdas)
        esperado_celdas = np.array(proba_h0_celdas) * n
        
        
        if ver:
            print("\nNew Cells:")
            print(celdas)
            print("")
            print("New expecteed values in cells")
            print(esperado_celdas)
        
            
    # ----------------------------------------------------------

    # tomar la ocurrencia de cada elemento del dominio
    ocurrencias_dominio = get_ocurrencias_dominio(sample, dominio)
    # obtener la ocurrido en cada celda
    observado_celdas = sumar_segun_celdas(ocurrencias_dominio, celdas)

    if ver:
        print("Observed values in cells")
        print(observado_celdas)
        
    # ----------------------------------------------------------

    # graficar si se quiere
    if ver:
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))

        # frecuencias
        frecuencias_por_valores = Counter(sample)
        valores = np.array(list(frecuencias_por_valores.keys()))
        frecuencias = np.array(list(frecuencias_por_valores.values()))

        # ordenar
        indices_ordenados = np.argsort(valores)
        valores = valores[indices_ordenados]
        frecuencias = frecuencias[indices_ordenados]

        # graficar
        x_plot = np.arange(0, max(sample) + 1)
        ax[0].bar(valores, frecuencias, label= "Oberved", color="blue")
        ax[0].plot(x_plot, pmf_estimada(x_plot)*n, color="red", label="Expected under H0")
        ax[0].set_title("Frequencies", fontsize = 18)
        ax[0].legend()

        # log scale
        ax[1].plot(x_plot, pmf_estimada(x_plot) * n, label= "Expected under H0",
                   color="red", linestyle="-")
        ax[1].plot(x_plot, get_ocurrencias_dominio(sample, x_plot), label="Observed",
                   color="blue", linestyle="", marker=".")
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_title("Frequencies (log scale)", fontsize = 18)
        ax[1].legend()

        # hacer los nombres de las celdas
        nombres_celdas = [str(c[0]) + "-" + str(c[1]) for c in celdas]

        # graficar celdas
        ax[2].bar(nombres_celdas, observado_celdas, label="Observed", color="blue")
        ax[2].plot(nombres_celdas, esperado_celdas, label="Expected under H0", color="red", marker="o")
        ax[2].set_yscale('log')
        ax[2].set_xlabel("Cells")
        ax[2].set_title("Data in cells", fontsize = 18)
        ax[2].tick_params(axis="x", rotation=90)
        ax[2].legend()
        
        plt.show()

    # ----------------------------------------------------------

    # comprobar que todo funcione
    assert np.isclose(sum(observado_celdas), sum(esperado_celdas))
    assert np.isclose(sum(observado_celdas), n)

    # hacer la prueba chi2
    T, pvalue = chisquare(f_obs = observado_celdas,
                          f_exp = esperado_celdas,
                          ddof = 3) # se estimaron 3 parametros

    # ver el resultado si se quiere
    if ver:
        print("\nChi square test \n")
        print(f"Statistic T = {T}")
        print(f"p-value = {pvalue}")
        
        # rechazar H0
        if pvalue <= 0.05:
            print("The distribution is NOT zero truncated power law")
        # aceptar H0
        else:
            print("The distribution is zero truncated power law")
        print("-"*100)
        
    return pvalue

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Zero discrete lognormal


# pmf sin nromalizar
def functional_zero_discrete_lognormal(d, mu, sigma):
    '''
    Funcional de la distribucion propuesta
    zero discrete lognormal
    
    Es decir, la pmf no normalizada para valores positivos
    '''
    
    # devolver para cuando d>0
    return np.power((d * sigma * np.sqrt(2 * np.pi)), -1) * np.exp(-np.power(np.log(d) - mu, 2) / (2 * np.power(sigma, 2)))


# log pmf sin nromalizar
def log_functional_zero_discrete_lognormal(d, mu, sigma):
    '''
    Logaritmo del funcional de la distribucion propuesta
    zero discrete lognormal
    
    Es decir, la pmf no normalizada para valores positivos, en nogaritmo
    '''
    
    # devolver para cuando d>0
    return -(np.log(d)  + np.log(sigma) + np.log(np.sqrt(2 * np.pi))) - np.power(np.log(d) - mu, 2) / (2 * np.power(sigma, 2))


# pmf 
def pmf_zero_discrete_lognormal(d, beta, mu, sigma,
                                C = None, D_max = 1000):
    '''
    Funcion de probabilidad de la distribucion propuesta
    zero discrete lognormal
    Parametros: beta, mu, sigma
    
    C es la constante normalizadora 
    (funcional evaluado en valores mayores a 0)
    
    D_max es el valor maximo considerado para d
    '''

    # poner las log probabilidades en un vector (si d es vector)
    resultado = np.zeros_like(d, float)
    
    # si no se tiene la constante normalizadora, calcularla
    if C is None:
        C = 1 / (functional_zero_discrete_lognormal(np.arange(1, D_max), mu, sigma)).sum()
        
    # donde es 0 poner beta
    resultado[d==0] = beta
    
    # donde d>0, poner el funcional normalizado
    resultado[d!=0] = (1 - beta) * C * functional_zero_discrete_lognormal(d[d!=0], mu, sigma)
    
    return resultado



# log pmf 
def log_pmf_zero_discrete_lognormal(d, beta, mu, sigma,
                                    C = None, D_max = 1000):
    '''
    Logaritmo de la funcion de probabilidad de la distribucion propuesta
    zero discrete lognormal
    Parametros: beta, mu, sigma
    
    C es la constante normalizadora 
    (funcional evaluado en valores mayores a 0)
    
    D_max es el valor maximo considerado para d
    '''

    # poner las log probabilidades en un vector (si d es vector)
    resultado = np.zeros_like(d, float)
    
    # si no se tiene la constante normalizadora, calcularla
    if C is None:
        C = 1 / (functional_zero_discrete_lognormal(np.arange(1, D_max), mu, sigma)).sum()
        
    # donde es 0 poner log beta
    resultado[d==0] = np.log(beta)
    
    # donde d>0, poner el funcional normalizado, todo en logaritmo
    resultado[d!=0] = np.log(1 - beta) + np.log(C) +  log_functional_zero_discrete_lognormal(d[d!=0], mu, sigma)
    
    return resultado




# estimar parametros
def estimate_zero_discrete_lognormal(sample, initial_point = [0.5, 0.5], D_max = 1000):
    '''
    Given a sample, estimate the parameters of the proposed distribution
    zero discrete lognormal
    Estimate: beta, mu, sigma
    '''
    sample = np.array(sample)
    
    # primero, beta se estima como el porcentaje de datos que son cero
    estimacion_beta = (sample == 0).sum()/len(sample)
    
    # tomar valores positivos de la muestra para estimar lo que sigue
    positive_sample = sample[sample>0]
    
    # minimizar el minus log likelihood para kappa y lambda
    
    # definir el minus_log_likelihood, como funcion de los parametros
    def compute_minus_log_likelihood(mu_sigma):
    
        # separar parametros
        mu = mu_sigma[0]
        sigma = mu_sigma[1]
    
        # calcular la constante de estos parametros
        C = 1 / (functional_zero_discrete_lognormal(np.arange(1, D_max), mu, sigma)).sum()
        
        # tomar la log probabilidad de cada elemento en la muestra
        log_proba = log_pmf_zero_discrete_lognormal(d = positive_sample,
                                                    beta = estimacion_beta,
                                                    mu = mu,
                                                    sigma = sigma, 
                                                    C = C, D_max = D_max)
        
        # devolver minus log likelihood
        return -log_proba.sum()
    
    
    # para estimar los parametros, poner un punto inicial
    punto_inicial = initial_point
    
    # limites de mu y sigma
    bounds = [(None, None), (1e-6, None)]
    
    # estimar los parametros como problema de minimizacion
    resultado = minimize(compute_minus_log_likelihood,
                         punto_inicial,
                         method ='nelder-mead',
                         bounds = bounds)
    
    # Obtener los resultados
    parametros_optimizados = resultado.x
    
    # separar los parametros
    estimacion_mu = parametros_optimizados[0]
    estimacion_sigma = parametros_optimizados[1]
    
    return {'beta': estimacion_beta,
           'mu': estimacion_mu,
           'sigma': estimacion_sigma}



# test for zero discrete lognormal
def analize_zero_discrete_lognormal(sample, D_max = 1000, initial_point = [0.5, 0.5],
                                    k_chi2 = 10, ver = True):
    '''
    Goodness of fit test for the proposed distriution
    Chi square goodness of fit test with k_chi2 cells
    zero discrete lognormal
    '''
    
    # ver el numero de datos
    n = len(sample)
    if ver:
        print(f"Sample of size {n}")

    # ----------------------------------------------------------

    # obtener laas estimaciones MLE de los parametros
    parametros_optimizados = estimate_zero_discrete_lognormal(sample, initial_point=initial_point)

    # separar
    estimacion_beta = parametros_optimizados['beta']
    estimacion_mu = parametros_optimizados['mu']
    estimacion_sigma = parametros_optimizados['sigma']

    # calcular la constante de normalizacion para estos parametros
    C = 1 / (functional_zero_discrete_lognormal(np.arange(1, D_max), estimacion_mu, estimacion_sigma)).sum()

    # show parameters
    if ver:
        print("\nEstimated parameters:")
        print(f"beta = {estimacion_beta}")
        print(f"mu = {estimacion_mu}")
        print(f"sigma = {estimacion_sigma}")
        print(f"Normalization C = {C}")


    # con estos parametros, delimitar
    # la funcion de probabilidad de H0
    def pmf_estimada(x):
        return pmf_zero_discrete_lognormal(x,
                                           beta = estimacion_beta,
                                           mu = estimacion_mu,
                                           sigma = estimacion_sigma,
                                           C = C, D_max = D_max)
        
    # comprobar que sea funcion de probabilidad en el dominio
    assert np.isclose(sum(pmf_estimada(np.arange(D_max))), 1)


    # ----------------------------------------------------------

    # el dominio lo delimita D_max
    dominio = np.arange(0, D_max)

    # obtener las probabilidades del dominio
    proba_h0_dominio = pmf_estimada(dominio)

    # ----------------------------------------------------------

    # tomar las celdas equiprobables
    celdas = obtener_k_celdas_equiprobables(proba_h0_dominio, k_chi2)
    if ver:
        print("\nCells:")
        print(celdas)
        print("")

    # ----------------------------------------------------------

    # tomar las probabilidades de la distribucion de H0 en las celdas (casi equiprobable)
    proba_h0_celdas = sumar_segun_celdas(proba_h0_dominio, celdas)
    # corregir en caso de haber errores
    proba_h0_celdas[-1] = 1 - sum(proba_h0_celdas[:-1])

    # ver que si sea probabilidad en las celdas
    assert comprobar_vector_probabilidad(proba_h0_celdas)

    # obtener lo esperado en cada celda
    # son solo las probabilidades por el numero de datos
    esperado_celdas = np.array(proba_h0_celdas) * n

    if ver:
        print("Expecteed values in cells")
        print(esperado_celdas)
        
        
    # ver que sean validas las celdas
    while not valid_expected(esperado_celdas):
        
        # bajar k en uno
        k_chi2 = k_chi2-1
        if ver:
            print("\nExpected values not valid")
            print(f"Reduce k to {k_chi2}")
        
        # volver a calcular
        celdas = obtener_k_celdas_equiprobables(proba_h0_dominio, k_chi2)
        
        # esperados
        proba_h0_celdas = sumar_segun_celdas(proba_h0_dominio, celdas)
        proba_h0_celdas[-1] = 1 - sum(proba_h0_celdas[:-1])
        assert comprobar_vector_probabilidad(proba_h0_celdas)
        esperado_celdas = np.array(proba_h0_celdas) * n
        
        
        if ver:
            print("\nNew Cells:")
            print(celdas)
            print("")
            print("New expecteed values in cells")
            print(esperado_celdas)
        
            
    # ----------------------------------------------------------

    # tomar la ocurrencia de cada elemento del dominio
    ocurrencias_dominio = get_ocurrencias_dominio(sample, dominio)
    # obtener la ocurrido en cada celda
    observado_celdas = sumar_segun_celdas(ocurrencias_dominio, celdas)

    if ver:
        print("Observed values in cells")
        print(observado_celdas)
        
    # ----------------------------------------------------------

    # graficar si se quiere
    if ver:
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))

        # frecuencias
        frecuencias_por_valores = Counter(sample)
        valores = np.array(list(frecuencias_por_valores.keys()))
        frecuencias = np.array(list(frecuencias_por_valores.values()))

        # ordenar
        indices_ordenados = np.argsort(valores)
        valores = valores[indices_ordenados]
        frecuencias = frecuencias[indices_ordenados]

        # graficar
        x_plot = np.arange(0, max(sample) + 1)
        ax[0].bar(valores, frecuencias, label= "Oberved", color="blue")
        ax[0].plot(x_plot, pmf_estimada(x_plot)*n, color="red", label="Expected under H0")
        ax[0].set_title("Frequencies", fontsize = 18)
        ax[0].legend()

        # log scale
        ax[1].plot(x_plot, pmf_estimada(x_plot) * n, label= "Expected under H0",
                   color="red", linestyle="-")
        ax[1].plot(x_plot, get_ocurrencias_dominio(sample, x_plot), label="Observed",
                   color="blue", linestyle="", marker=".")
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_title("Frequencies (log scale)", fontsize = 18)
        ax[1].legend()

        # hacer los nombres de las celdas
        nombres_celdas = [str(c[0]) + "-" + str(c[1]) for c in celdas]

        # graficar celdas
        ax[2].bar(nombres_celdas, observado_celdas, label="Observed", color="blue")
        ax[2].plot(nombres_celdas, esperado_celdas, label="Expected under H0", color="red", marker="o")
        ax[2].set_yscale('log')
        ax[2].set_xlabel("Cells")
        ax[2].set_title("Data in cells", fontsize = 18)
        ax[2].tick_params(axis="x", rotation=90)
        ax[2].legend()
        
        plt.show()


    # ----------------------------------------------------------

    # comprobar que todo funcione
    assert np.isclose(sum(observado_celdas), sum(esperado_celdas))
    assert np.isclose(sum(observado_celdas), n)

    # hacer la prueba chi2
    T, pvalue = chisquare(f_obs = observado_celdas,
                          f_exp = esperado_celdas,
                          ddof = 3) # se estimaron 3 parametros

    # ver el resultado si se quiere
    if ver:
        print("\nChi square test \n")
        print(f"Statistic T = {T}")
        print(f"p-value = {pvalue}")
        
        # rechazar H0
        if pvalue <= 0.05:
            print("The distribution is NOT zero discrete lognormal")
        # aceptar H0
        else:
            print("The distribution is zero discrete lognormal")
        print("-"*100)
        
    return pvalue



# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
















