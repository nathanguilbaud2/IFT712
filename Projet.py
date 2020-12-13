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
    
    print("\nx_entrainement[0] = ",x_entrainement[0],"\n")
    print("t_entrainement[0] = ",t_entrainement[0],"\n")
    print("x_test[0] = ",x_test[0],"\n")

if __name__ == "__main__":
    main()
