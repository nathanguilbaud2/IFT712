import numpy as np
import pandas as pd


class GestionDonnees:
    """
    Fonction qui génère les données d'entrainement et de test
    retourne les données d'entrainement, leur classe et les données de test
    """ 
    def generer_donnees(self):
        entrainement_csv = pd.read_csv("Donnees/train.csv")
        
        rows, cols = (len(entrainement_csv), 64*3) 
        x_train = [[0 for i in range(cols)] for j in range(rows)] 
        t_train=[0]*len(entrainement_csv)
        
        for line in range(len(entrainement_csv)):
            t_train[line] = entrainement_csv.loc[line,'species']
            for i in range(1,65):
                x_train[line][i] = entrainement_csv.loc[line,'margin'+str(i)]
            for i in range(1,65):
                x_train[line][i+63]= entrainement_csv.loc[line,'shape'+str(i)]
            for i in range(1,65):
                x_train[line][i+127]= entrainement_csv.loc[line,'texture'+str(i)]
        
        test_csv = pd.read_csv("Donnees/test.csv")
        
        rows, cols = (len(test_csv), 64*3+1) 
        x_test = [[9 for i in range(cols)] for j in range(rows)] 
        
        for line in range(len(test_csv)):
            x_test[line][0] = test_csv.loc[line,'id']
            for i in range(1,65):
                x_test[line][i] = test_csv.loc[line,'margin'+str(i)]
            for i in range(1,65):
                x_test[line][i+64]= test_csv.loc[line,'shape'+str(i)]
            for i in range(1,65):
                x_test[line][i+128]= test_csv.loc[line,'texture'+str(i)]
        return x_train,t_train,x_test


