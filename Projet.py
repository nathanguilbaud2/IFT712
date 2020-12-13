# -*- coding: utf-8 -*-

import numpy as np
import gestion_donnees as gd
import time
import sys

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
def main(): 
    if len(sys.argv) < 3:
        usage = "\n Usage: python All_2.py choix_algorithme recherche_parametres\
        \n\n\t choix_algorithme:\
        \n\t\t -1 : Tous les algorithmes\
        \n\t\t 0 : Gradient Boosting Classifier\
        \n\t\t 1 : Random-Forest\
        \n\t\t 2 : ADA-Boost\
        \n\t\t 3 : Decision-Tree\
        \n\t\t 4 : SVM\
        \n\t\t 5 : K-nearest neighbors\
        \n\t\t 6 : Linear Discriminant Analysis\
        \n\t\t 7 : Quadratic Discriminant Analysis\
        \n\t recherche_parametres :\
        \n\t\t 0 : Pas de recherche de parametres\
        \n\t\t 1 : Recherche des meilleurs parametres\n"
        print(usage)
        return

    choix_algorithme = int(sys.argv[1])
    recherche_parametres = int(sys.argv[2])
    
    # On génère les données d'entraînement et de test
    generateur_donnees = gd.GestionDonnees()
    start = time.time()
    [x_entrainement,t_entrainement,x_test] = generateur_donnees.generer_donnees()
    print("------- Fin du tri des donnees en ", time.time()-start, 'seconds')
    
    # Nous allons étudier 8 algorithmes différents
    liste_variables=[[[0] for j in range(len(x_entrainement))] for j in range(7)]    
    liste_variables_test=[[[0] for j in range(len(x_test))] for j in range(7)]
    
    # Liste des parametres pour chaque algorithmes
    liste_parametres_atester=[     [[1,10,100],[1,0.1,0.01]]   ,    [[100,537,1000],[None,10]]    ,    [[100,1000],[1,0.1,0.01]]    ,    [[6,20,100],[None,10,100]]    ,    [[0.5,1,1.5,10,50,100,200,500,1000,5000,10000],["linear", "poly", "rbf", "sigmoid"]]    ,    [[1,5,8,10],["ball_tree", "kd_tree", "brute"]]    ,    [["svd","lsqr"],[1,0.1,0.01,0.001,0.0001]]   ,    [[100,1000,10000],[1,10,100]]    ]
    liste_meilleurs_parametres=[ [1,1] , [100,None] , [100,1] , [6,1] , [1,"linear"] , [1,"ball_tree"] , ["svd",1] , [100,10] ]

    meilleurs_parametres=[[],[],[],[],[],[],[],[]]
    
    liste_variables_label=["Utilisation de Margin seulement","Utilisation de Shape seulement","Utilisation de Texture seulement","Utilisation de Margin et Shape","Utilisation de Margin et Texture","Utilisation de Shape et Texture","Utilisation de toutes les variables"]
    list_Algorithme_label=["~~~~~~~~~~~~~ GradientBoostingClassifier ~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~~~ Random-Forest ~~~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~~~~~ ADA-Boost ~~~~~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~~~~ Decision-Tree ~~~~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~ SVM ~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~~~~ K-nearest neighbors ~~~~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~ Linear Discriminant Analysis ~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~ Bagging Classifier ~~~~~~~~~~~~~~~~~~"]
    
    for i in range(len(x_entrainement)):
        liste_variables[0][i]=x_entrainement[i][0:64] # Seulement Margin
        liste_variables[1][i]=x_entrainement[i][64:128] # Seulement Shape
        liste_variables[2][i]=x_entrainement[i][128:] # Seulement Texture
       
        liste_variables[3][i]=x_entrainement[i][0:128] # Seulement Margin and Shape
        liste_variables[4][i]=np.concatenate((x_entrainement[i][0:64], x_entrainement[i][128:]), axis=0) # Seulement Margin and Texture
        liste_variables[5][i]=x_entrainement[i][64:] # Seulement Shape and Texture
        
        liste_variables[6][i]=x_entrainement[i] # Tous les parametres
        
    for i in range(len(x_test)):
        liste_variables_test[0][i]=x_test[i][1:65] # Seulement Margin
        liste_variables_test[1][i]=x_test[i][65:129] # Seulement Shape
        liste_variables_test[2][i]=x_test[i][129:] # Seulement Texture
       
        liste_variables_test[3][i]=x_test[i][1:129] # Seulement Margin and Shape
        liste_variables_test[4][i]=np.concatenate((x_test[i][1:65], x_test[i][129:]), axis=0) # Seulement Margin and Texture
        liste_variables_test[5][i]=x_test[i][65:] # Seulement Shape and Texture
        
        liste_variables_test[6][i]=x_test[i][1:] # Tous les parametres
       
    if(choix_algorithme==-1): # Pour tous les algorithmes
        for algorithme_choisi in range(len(list_Algorithme_label)):
            print("\n",list_Algorithme_label[algorithme_choisi])
            if(recherche_parametres==1): # On recherche les meilleurs hyperparametres
                print("\n           Avec recherche d'hypermarametres")
                meilleurs_parametres[algorithme_choisi] = Recherche_meilleurs_parametres(liste_variables[6], t_entrainement, liste_parametres_atester[algorithme_choisi], algorithme_choisi)
                print("Les meilleurs parametres calculés sont : ", meilleurs_parametres[algorithme_choisi][0], " et ", meilleurs_parametres[algorithme_choisi][1],"\n")
            else: # On prend les meilleurs parametres
                print("\n           Sans recherche d'hypermarametres")
                meilleurs_parametres[algorithme_choisi] = [liste_meilleurs_parametres[algorithme_choisi][0],liste_meilleurs_parametres[algorithme_choisi][1]]
                print("Les meilleurs parametres préalablement calculés sont : ", meilleurs_parametres[0], " et ", meilleurs_parametres[1],"\n")
            for variable_choisie in range(len(liste_variables)):
                Run_algorithme(liste_variables_label[variable_choisie], liste_variables[variable_choisie], t_entrainement, meilleurs_parametres[algorithme_choisi], algorithme_choisi)

    else: # Pour un algorithme
        print("\n",list_Algorithme_label[choix_algorithme])
        if(recherche_parametres==1): # On recherche les meilleurs hyperparametres
            print("\n           Avec recherche d'hypermarametres")
            meilleurs_parametres[choix_algorithme] = Recherche_meilleurs_parametres(liste_variables[6], t_entrainement, liste_parametres_atester[choix_algorithme], choix_algorithme)
            print("Les meilleurs parametres calculés sont : ", meilleurs_parametres[choix_algorithme][0], " et ", meilleurs_parametres[choix_algorithme][1],"\n")
        else: # On prend les meilleurs parametres
            print("\n           Sans recherche d'hypermarametres")
            meilleurs_parametres[choix_algorithme] = [liste_meilleurs_parametres[choix_algorithme][0],liste_meilleurs_parametres[choix_algorithme][1]]
            print("Les meilleurs parametres préalablement calculés sont : ", meilleurs_parametres[choix_algorithme][0], " et ", meilleurs_parametres[choix_algorithme][1],"\n")
            
        for variable_choisie in range(len(liste_variables)):
            Run_algorithme(liste_variables_label[variable_choisie], liste_variables[variable_choisie], t_entrainement, meilleurs_parametres[choix_algorithme], choix_algorithme)
    
"""
Fonction qui entraine un modele en fontion de l'algorithme choisi
""" 
def Run_algorithme(label, x_entrainement, t_entrainement, liste_parametres_algorithme ,numero_algorithme):
    print(label)
    # Choix de l'algorithme a utiliser
    if(numero_algorithme==0): # 0 = Gradient Boosting Classifier
        clf = GradientBoostingClassifier(n_estimators=liste_parametres_algorithme[0], learning_rate=liste_parametres_algorithme[1], random_state=0)

    elif(numero_algorithme==1): # 1 = Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=liste_parametres_algorithme[0], max_depth=liste_parametres_algorithme[1], random_state=0)

    elif(numero_algorithme==2): # 2 = ADA-Boost Classifier
        clf = AdaBoostClassifier(n_estimators=liste_parametres_algorithme[0], learning_rate=liste_parametres_algorithme[1], random_state=0)

    elif(numero_algorithme==3): # 3 = Decision Tree Classifier
        clf = tree.DecisionTreeClassifier(min_samples_split=liste_parametres_algorithme[0], max_depth=liste_parametres_algorithme[1], random_state=0)

    elif(numero_algorithme==4): # 4 = SVM
        clf = svm.SVC(C=liste_parametres_algorithme[0], kernel=liste_parametres_algorithme[1], probability=True)

    elif(numero_algorithme==5): # 5 = K-Nearest Neighbors
        clf = KNeighborsClassifier(n_neighbors=liste_parametres_algorithme[0], algorithm=liste_parametres_algorithme[1])

    elif(numero_algorithme==6): # 6 = Linear Discriminant Analysis
        clf = LinearDiscriminantAnalysis(solver=liste_parametres_algorithme[0], tol=liste_parametres_algorithme[1])

    elif(numero_algorithme==7): # 7 = Quadratic Discriminant Analysis
        clf = BaggingClassifier(n_estimators=liste_parametres_algorithme[0], max_samples=liste_parametres_algorithme[1])

    # Validation croisee #
    precision_entrainement=0
    precision_test=0
    K=10
    for i in range(0,K):
        xk_entrainement = np.delete(x_entrainement, slice(int(i*(len(x_entrainement)/K)), int(i*(len(x_entrainement)/K)+(len(x_entrainement)/K))), axis=0)
        tk_entrainement = np.delete(t_entrainement, slice(int(i*(len(t_entrainement)/K)), int(i*(len(t_entrainement)/K)+(len(t_entrainement)/K))), axis=0)
        xk_test = x_entrainement[int(i*(len(x_entrainement)/K)):int(i*(len(x_entrainement)/K)+(len(x_entrainement)/K))]
        tk_test = t_entrainement[int(i*(len(t_entrainement)/K)):int(i*(len(t_entrainement)/K)+(len(t_entrainement)/K))]
        
        clf.fit(xk_entrainement, tk_entrainement) # Entrainement
        
        precision_entrainement = precision_entrainement + clf.score(xk_entrainement, tk_entrainement)*100 # Precision sur les données d'entrainement
        precision_test = precision_test + clf.score(xk_test, tk_test)*100 # Precision sur les données de test
        
    precision_entrainement = precision_entrainement / K
    precision_test = precision_test / K
        
    print("Précision données d'entrainement : " , precision_entrainement, " %")
    print("Précision données de test : " , precision_test, " %\n")
    
    #return precision_test # On retourne la moyenne de la precision des donnees de test sur les 10 entrainements
 
    
"""
Fonction qui recherche les 2 meilleurs parametres pour l'algorithme passé en parametres
retourne les 2 meilleurs parametres pour l'algorithme passé en parametres
"""    
def Recherche_meilleurs_parametres(x_entrainement, t_entrainement, liste_parametres, numero_algorithme):
    meilleur_premier_parametre=0
    meilleur_second_parametre=0
    maximum = 0
    if(numero_algorithme==0): # 0 = Gradient Boosting Classifier
        for n_estimators in liste_parametres[0]:
            for learning_rate in liste_parametres[1]:
                clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
                clf.fit(x_entrainement[0:900], t_entrainement[0:900])
            
                if(clf.score(x_entrainement[900:990],t_entrainement[900:990])*100 > maximum):
                    maximum = clf.score(x_entrainement[900:990],t_entrainement[900:990])*100
                    meilleur_premier_parametre=n_estimators
                    meilleur_second_parametre=learning_rate
      
    elif(numero_algorithme==1): # 1 = Random-Forest Classifier
        for n_estimators in liste_parametres[0]:
            for max_depth in liste_parametres[1]:
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
                clf.fit(x_entrainement[0:900], t_entrainement[0:900])
            
                if(clf.score(x_entrainement[900:990],t_entrainement[900:990])*100 > maximum):
                    maximum = clf.score(x_entrainement[900:990],t_entrainement[900:990])*100
                    meilleur_premier_parametre=n_estimators
                    meilleur_second_parametre=max_depth
 
    elif(numero_algorithme==2): # 2 = ADA-Boost Classifier
        for n_estimators in liste_parametres[0]:
            for learning_rate in liste_parametres[1]:
                clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
                clf.fit(x_entrainement[0:900], t_entrainement[0:900])
            
                if(clf.score(x_entrainement[900:990],t_entrainement[900:990])*100 > maximum):
                    maximum = clf.score(x_entrainement[900:990],t_entrainement[900:990])*100
                    meilleur_premier_parametre=n_estimators
                    meilleur_second_parametre=learning_rate

    elif(numero_algorithme==3): # 3 = Decision Tree Classifier
        for min_samples_split in liste_parametres[0]:
            for max_depth in liste_parametres[1]:
                clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth, random_state=0)
                clf.fit(x_entrainement[0:900], t_entrainement[0:900])
            
                if(clf.score(x_entrainement[900:990],t_entrainement[900:990])*100 > maximum):
                    maximum = clf.score(x_entrainement[900:990],t_entrainement[900:990])*100
                    meilleur_premier_parametre=min_samples_split
                    meilleur_second_parametre=max_depth
      
    elif(numero_algorithme==4): # 4 = SVM
        for C in liste_parametres[0]:
            for kernel in liste_parametres[1]:
                clf = svm.SVC(C=C, kernel=kernel, probability=True)
                clf.fit(x_entrainement[0:900], t_entrainement[0:900])
            
                if(clf.score(x_entrainement[900:990],t_entrainement[900:990])*100 > maximum):
                    maximum = clf.score(x_entrainement[900:990],t_entrainement[900:990])*100
                    meilleur_premier_parametre=C
                    meilleur_second_parametre=kernel
        
    elif(numero_algorithme==5): # 5 = K-Nearest Neighbors
        for n_neighbors in liste_parametres[0]:
            for algorithm in liste_parametres[1]:
                clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
                clf.fit(x_entrainement[0:900], t_entrainement[0:900])
            
                if(clf.score(x_entrainement[900:990],t_entrainement[900:990])*100 > maximum):
                    maximum = clf.score(x_entrainement[900:990],t_entrainement[900:990])*100
                    meilleur_premier_parametre=n_neighbors
                    meilleur_second_parametre=algorithm

    elif(numero_algorithme==6): # 6 = Linear Discriminant Analysis
        for solver in liste_parametres[0]:
            for tol in liste_parametres[1]:
                clf = LinearDiscriminantAnalysis(solver=solver, tol=tol)
                clf.fit(x_entrainement[0:900], t_entrainement[0:900])
            
                if(clf.score(x_entrainement[900:990],t_entrainement[900:990])*100 > maximum):
                    maximum = clf.score(x_entrainement[900:990],t_entrainement[900:990])*100
                    meilleur_premier_parametre=solver
                    meilleur_second_parametre=tol

    elif(numero_algorithme==7): # 7 = Quadradic Discriminant Analysis
        for n_estimators in liste_parametres[0]:
            for max_samples in liste_parametres[1]:
                clf = BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples)
                clf.fit(x_entrainement[0:900], t_entrainement[0:900])
            
                if(clf.score(x_entrainement[900:990],t_entrainement[900:990])*100 > maximum):
                    maximum = clf.score(x_entrainement[900:990],t_entrainement[900:990])*100
                    meilleur_premier_parametre=n_estimators
                    meilleur_second_parametre=max_samples

    return meilleur_premier_parametre,meilleur_second_parametre 

if __name__ == "__main__":
    main()
