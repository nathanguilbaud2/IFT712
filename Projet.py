# -*- coding: utf-8 -*-

import numpy as np
import gestion_donnees as gd
import time

def main(): 
    
    # On génère les données d'entraînement et de test
    generateur_donnees = gd.GestionDonnees()
    start = time.time()
    [x_entrainement,t_entrainement,x_test] = generateur_donnees.generer_donnees()
    print("------- Fin du tri des donnees en ", time.time()-start, 'seconds')
    
    # Nous allons étudier 8 algorithmes différents
    liste_variables=[[[0] for j in range(len(x_entrainement))] for j in range(8)]    
    liste_variables_test=[[[0] for j in range(len(x_test))] for j in range(8)]
    test_id=[[0] for j in range(len(x_test))]
    
    # Séparation des données d'entrainement en fonction du nombre de variables choisi
    for i in range(len(x_entrainement)):
        liste_variables[0][i]=x_entrainement[i][0:64] # Seulement Margin
        liste_variables[1][i]=x_entrainement[i][64:128] # Seulement Shape
        liste_variables[2][i]=x_entrainement[i][128:] # Seulement Texture
       
        liste_variables[3][i]=x_entrainement[i][0:128] # Seulement Margin and Shape
        liste_variables[4][i]=np.concatenate((x_entrainement[i][0:64], x_entrainement[i][128:]), axis=0) # Seulement Margin and Texture
        liste_variables[5][i]=x_entrainement[i][64:] # Seulement Shape and Texture
        
        liste_variables[6][i]=x_entrainement[i] # Tous les parametres
    
    # Séparation des données de test en fonction du nombre de variables choisi
    for i in range(len(x_test)):
        test_id[i] = x_test[i][0]
        liste_variables_test[0][i]=x_test[i][1:65] # Seulement Margin
        liste_variables_test[1][i]=x_test[i][65:129] # Seulement Shape
        liste_variables_test[2][i]=x_test[i][129:] # Seulement Texture
       
        liste_variables_test[3][i]=x_test[i][1:129] # Seulement Margin and Shape
        liste_variables_test[4][i]=np.concatenate((x_test[i][1:65], x_test[i][129:]), axis=0) # Seulement Margin and Texture
        liste_variables_test[5][i]=x_test[i][65:] # Seulement Shape and Texture
        
        liste_variables_test[6][i]=x_test[i][1:] # Tous les parametres

if __name__ == "__main__":
    main()
