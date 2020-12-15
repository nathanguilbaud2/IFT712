# -*- coding: utf-8 -*-

import numpy as np
import sys
import gestion_donnees as gd
import time
import pickle
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier

"""
Liste des algorithmes utilisés et leur numéro correspondant

0 : Gradient Boosting Classifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier.predict)
1 : Random-Forest (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
2 : ADA-Boost (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
3 : Decision-Tree (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
4 : SVM (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
5 : K-nearest neighbors (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
6 : Linear Discriminant Analysis (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
7 : Quadratic Discriminant Analysis (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)
"""

def main(): 
    if(len(sys.argv) != 4):
        usage = "\n Usage: python Projet.py choix_algorithme recherche_parametres generer_soumission\
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
        \n\t\t 1 : Recherche des meilleurs parametres\
        \n\t generer_soumission :\
        \n\t\t 0 : Pas de prédiction sur les données du fichier test.csv\
        \n\t\t 1 : Prédiction sur les données du fichier test.csv\n"
        print(usage)
        return

    choix_algorithme = int(sys.argv[1])
    recherche_parametres = int(sys.argv[2])
    generer_soumission = int(sys.argv[3])
    
    # On génère les données d'entraînement et de test
    generateur_donnees = gd.GestionDonnees()
    start = time.time()
    [x_entrainement,t_entrainement,x_test] = generateur_donnees.generer_donnees()
    print("------- Fin du tri des donnees en ", time.time()-start, 'seconds')

    # Nous allons étudier 8 algorithmes différents
    liste_variables=[[[0] for j in range(len(x_entrainement))] for j in range(7)]    
    liste_variables_test=[[[0] for j in range(len(x_test))] for j in range(7)]
    test_id=[[0] for j in range(len(x_test))]   
    
    # Liste des hyperparametres à tester pour chaque fonction
    liste_parametres_atester=[     [[1,10,100],[1,0.1,0.01]]   ,    [[100,537,1000],[None,10]]    ,    [[100,1000],[1,0.1,0.01]]    ,    [[6,20,100],[None,10,100]]    ,    [[0.5,1,1.5,10,50,100,200,500,1000,5000,10000],["linear", "poly", "rbf", "sigmoid"]]    ,    [[1,5,8,10],["ball_tree", "kd_tree", "brute"]]    ,    [["svd","lsqr"],[1,0.1,0.01,0.001,0.0001]]   ,    [[100,1000,10000],[1,10,100]]    ]    
    # Liste des meilleurs hyperparametres de chaque fonction
    liste_meilleurs_parametres=[ [100,0.01] , [1000,None] , [1000,0.01] , [6,None] , [200,"linear"] , [1,"ball_tree"] , ["svd",0.1] , [10000,100] ]
    
    meilleurs_parametres=[[],[],[],[],[],[],[],[]]
    meilleur_resultat=[[0] for j in range(len(liste_parametres_atester))]
    meilleur_resultat_variables=[[0] for j in range(len(liste_parametres_atester))]
    
    # Labels
    liste_variables_label=["Utilisation de Margin seulement","Utilisation de Shape seulement","Utilisation de Texture seulement","Utilisation de Margin et Shape","Utilisation de Margin et Texture","Utilisation de Shape et Texture","Utilisation de toutes les variables"]
    liste_variables_fichier=["Margin","Shape","Texture","Margin_Shape","Margin_Texture","Shape_Texture","Tout"]
    list_Algorithme_label=["~~~~~~~~~~~~~ GradientBoostingClassifier ~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~~~ Random-Forest ~~~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~~~~~ ADA-Boost ~~~~~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~~~~ Decision-Tree ~~~~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~ SVM ~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~~~~ K-nearest neighbors ~~~~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~ Linear Discriminant Analysis ~~~~~~~~~~~~~~~~~~","~~~~~~~~~~~~~~~~~~ Bagging Classifier ~~~~~~~~~~~~~~~~~~"]
    Test_resultat=["GradientBoostingClassifier","Random-Forest","ADA-Boost","Decision-Tree","SVM","K-nearestneighbors","Linear_Discriminant_Analysis","BaggingClassifier"]
    
    for i in range(len(x_entrainement)):
        liste_variables[0][i]=x_entrainement[i][0:64] # Seulement Margin
        liste_variables[1][i]=x_entrainement[i][64:128] # Seulement Shape
        liste_variables[2][i]=x_entrainement[i][128:] # Seulement Texture
       
        liste_variables[3][i]=x_entrainement[i][0:128] # Seulement Margin and Shape
        liste_variables[4][i]=np.concatenate((x_entrainement[i][0:64], x_entrainement[i][128:]), axis=0) # Seulement Margin and Texture
        liste_variables[5][i]=x_entrainement[i][64:] # Seulement Shape and Texture
        
        liste_variables[6][i]=x_entrainement[i] # Tous les parametres
        
    for i in range(len(x_test)):
        test_id[i] = x_test[i][0]
        liste_variables_test[0][i]=x_test[i][1:65] # Seulement Margin
        liste_variables_test[1][i]=x_test[i][65:129] # Seulement Shape
        liste_variables_test[2][i]=x_test[i][129:] # Seulement Texture
       
        liste_variables_test[3][i]=x_test[i][1:129] # Seulement Margin and Shape
        liste_variables_test[4][i]=np.concatenate((x_test[i][1:65], x_test[i][129:]), axis=0) # Seulement Margin and Texture
        liste_variables_test[5][i]=x_test[i][65:] # Seulement Shape and Texture
        
        liste_variables_test[6][i]=x_test[i][1:] # Tous les parametres
     
    # Si on sélectionne tous les algorithmes
    if(choix_algorithme==-1):
        for algorithme_choisi in range(len(list_Algorithme_label)):
            print("\n",list_Algorithme_label[algorithme_choisi])
            
            # Recherche des meilleurs parametres de l'algorithme
            if(recherche_parametres==1): # On recherche les meilleurs hyperparametres
                print("\n           Avec recherche d'hypermarametres")
                meilleurs_parametres[algorithme_choisi] = Recherche_meilleurs_parametres(liste_variables[6], t_entrainement, liste_parametres_atester[algorithme_choisi], algorithme_choisi)
                print("Les meilleurs parametres calculés sont : ", meilleurs_parametres[algorithme_choisi][0], " et ", meilleurs_parametres[algorithme_choisi][1],"\n")
            
            # Pas de recherche des meilleurs parametres de l'algorithme
            else: # On prend les meilleurs parametres
                print("\n           Sans recherche d'hypermarametres")
                meilleurs_parametres[algorithme_choisi] = [liste_meilleurs_parametres[algorithme_choisi][0],liste_meilleurs_parametres[algorithme_choisi][1]]
                print("Les meilleurs parametres préalablement calculés sont : ", meilleurs_parametres[algorithme_choisi][0], " et ", meilleurs_parametres[algorithme_choisi][1],"\n")
                          
            meilleur_resultat[algorithme_choisi]=0
            meilleur_resultat_variables[algorithme_choisi]=""
            for variable_choisie in range(len(liste_variables)):
                precision = Run_algorithme(liste_variables_label[variable_choisie],liste_variables[variable_choisie],t_entrainement,liste_variables_test[variable_choisie],meilleurs_parametres[algorithme_choisi],algorithme_choisi,Test_resultat[algorithme_choisi],generer_soumission,liste_variables_fichier[variable_choisie],test_id)
                if(precision > meilleur_resultat[algorithme_choisi]):
                    meilleur_resultat[algorithme_choisi] = precision
                    meilleur_resultat_variables[algorithme_choisi]=liste_variables_label[algorithme_choisi]
        
        # Visualisation graphique des meilleurs résultats de chaque algorithme
        print("\n")
        for resultat in range(len(meilleur_resultat)):
            print(Test_resultat[resultat], " : ",meilleur_resultat[resultat])
        print("\n")
        plt.figure(0)
        plt.bar(Test_resultat, meilleur_resultat)
        plt.show()  
        
    # Si on sélectionne un algorithme
    else: # Pour un algorithme
        print("\n",list_Algorithme_label[choix_algorithme])
          
        # Recherche des meilleurs parametres de l'algorithme
        if(recherche_parametres==1):
            print("\n           Avec recherche d'hypermarametres")
            meilleurs_parametres[choix_algorithme] = Recherche_meilleurs_parametres(liste_variables[6], t_entrainement, liste_parametres_atester[choix_algorithme], choix_algorithme)
            print("Les meilleurs parametres calculés sont : ", meilleurs_parametres[choix_algorithme][0], " et ", meilleurs_parametres[choix_algorithme][1],"\n")

        # Pas de recherche des meilleurs parametres de l'algorithme
        else:
            print("\n           Sans recherche d'hypermarametres")
            meilleurs_parametres[choix_algorithme] = [liste_meilleurs_parametres[choix_algorithme][0],liste_meilleurs_parametres[choix_algorithme][1]]
            print("Les meilleurs parametres préalablement calculés sont : ", meilleurs_parametres[choix_algorithme][0], " et ", meilleurs_parametres[choix_algorithme][1],"\n")

        for variable_choisie in range(len(liste_variables)):
            Run_algorithme(liste_variables_label[variable_choisie],liste_variables[variable_choisie],t_entrainement,liste_variables_test[variable_choisie],meilleurs_parametres[choix_algorithme],choix_algorithme,Test_resultat[choix_algorithme],generer_soumission,liste_variables_fichier[variable_choisie],test_id)

    
"""
Fonction qui entraine un modele
retourne la précision du modele sur les données de test
""" 
def Run_algorithme(label, x_entrainement, t_entrainement, x_test, liste_parametres, numero_algorithme,Test_result,generer_soumission,liste_variables_fichier,test_id):
    print(label)
    # Choix de l'algorithme a utiliser
    if(numero_algorithme==0): # 0 = Gradient Boosting Classifier
        clf = GradientBoostingClassifier(n_estimators=liste_parametres[0], learning_rate=liste_parametres[1], random_state=0)

    elif(numero_algorithme==1): # 1 = Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=liste_parametres[0], max_depth=liste_parametres[1], random_state=0)

    elif(numero_algorithme==2): # 2 = ADA-Boost Classifier
        clf = AdaBoostClassifier(n_estimators=liste_parametres[0], learning_rate=liste_parametres[1], random_state=0)

    elif(numero_algorithme==3): # 3 = Decision Tree Classifier
        clf = tree.DecisionTreeClassifier(min_samples_split=liste_parametres[0], max_depth=liste_parametres[1], random_state=0)

    elif(numero_algorithme==4): # 4 = SVM
        clf = svm.SVC(C=liste_parametres[0], kernel=liste_parametres[1], probability=True)

    elif(numero_algorithme==5): # 5 = K-Nearest Neighbors
        clf = KNeighborsClassifier(n_neighbors=liste_parametres[0], algorithm=liste_parametres[1])

    elif(numero_algorithme==6): # 6 = Linear Discriminant Analysis
        clf = LinearDiscriminantAnalysis(solver=liste_parametres[0], tol=liste_parametres[1])

    elif(numero_algorithme==7): # 7 = Quadratic Discriminant Analysis
        clf = BaggingClassifier(n_estimators=liste_parametres[0], max_samples=liste_parametres[1])

    # Validation croisee #
    [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement) 
        
    print("Précision données d'entrainement : " , precision_entrainement, " %")
    print("Précision données de test : " , precision_test, " %\n")
    
    #Entrainement du modèle sur toutes les données d'entrainement
    clf.fit(x_entrainement,t_entrainement) 
    
    # Si on souhaite génerer un fichier CSV de prédiction des données du fichier test.csv 
    # Cette partie permet de faire des prédictions sur des données que le modèle n'a jamais vu et qui sont nécessaires pour la compétition Kaggle
    if(generer_soumission==1):
        with open("Submission/Submission_"+Test_result+'/'+Test_result+'_'+liste_variables_fichier+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            
            # On récupère la liste des colonnes de prediction (le nom des feuilles)
            liste_nom_feuilles=[[''] for j in range(len(clf.classes_)+1)]
            liste_nom_feuilles[0]='id'
            for i in range(len(clf.classes_)):
                liste_nom_feuilles[i+1]=clf.classes_[i]
            # On écrit le header du fichier CSV
            spamwriter.writerow(liste_nom_feuilles)
            
            # On écrit la prédiction de chaque données tests pour chaque classes de sorties
            aray_resultats_proba=clf.predict_proba(x_test)
            for a in range(len(aray_resultats_proba)):
                proba = [int(test_id[a]),aray_resultats_proba[a][0],aray_resultats_proba[a][1],aray_resultats_proba[a][2],aray_resultats_proba[a][3],aray_resultats_proba[a][4],aray_resultats_proba[a][5],aray_resultats_proba[a][6],aray_resultats_proba[a][7],aray_resultats_proba[a][8],aray_resultats_proba[a][9],aray_resultats_proba[a][10],aray_resultats_proba[a][11],aray_resultats_proba[a][12],aray_resultats_proba[a][13],aray_resultats_proba[a][14],aray_resultats_proba[a][15],aray_resultats_proba[a][16],aray_resultats_proba[a][17],aray_resultats_proba[a][18],aray_resultats_proba[a][19],aray_resultats_proba[a][20],aray_resultats_proba[a][21],aray_resultats_proba[a][22],aray_resultats_proba[a][23],aray_resultats_proba[a][24],aray_resultats_proba[a][25],aray_resultats_proba[a][26],aray_resultats_proba[a][27],aray_resultats_proba[a][28],aray_resultats_proba[a][29],aray_resultats_proba[a][30],aray_resultats_proba[a][31],aray_resultats_proba[a][32],aray_resultats_proba[a][33],aray_resultats_proba[a][34],aray_resultats_proba[a][35],aray_resultats_proba[a][36],aray_resultats_proba[a][37],aray_resultats_proba[a][38],aray_resultats_proba[a][39],aray_resultats_proba[a][40],aray_resultats_proba[a][41],aray_resultats_proba[a][42],aray_resultats_proba[a][43],aray_resultats_proba[a][44],aray_resultats_proba[a][45],aray_resultats_proba[a][46],aray_resultats_proba[a][47],aray_resultats_proba[a][48],aray_resultats_proba[a][49],aray_resultats_proba[a][50],aray_resultats_proba[a][51],aray_resultats_proba[a][52],aray_resultats_proba[a][53],aray_resultats_proba[a][54],aray_resultats_proba[a][55],aray_resultats_proba[a][56],aray_resultats_proba[a][57],aray_resultats_proba[a][58],aray_resultats_proba[a][59],aray_resultats_proba[a][60],aray_resultats_proba[a][61],aray_resultats_proba[a][62],aray_resultats_proba[a][63],aray_resultats_proba[a][64],aray_resultats_proba[a][65],aray_resultats_proba[a][66],aray_resultats_proba[a][67],aray_resultats_proba[a][68],aray_resultats_proba[a][69],aray_resultats_proba[a][70],aray_resultats_proba[a][71],aray_resultats_proba[a][72],aray_resultats_proba[a][73],aray_resultats_proba[a][74],aray_resultats_proba[a][75],aray_resultats_proba[a][76],aray_resultats_proba[a][77],aray_resultats_proba[a][78],aray_resultats_proba[a][79],aray_resultats_proba[a][80],aray_resultats_proba[a][81],aray_resultats_proba[a][82],aray_resultats_proba[a][83],aray_resultats_proba[a][84],aray_resultats_proba[a][85],aray_resultats_proba[a][86],aray_resultats_proba[a][87],aray_resultats_proba[a][88],aray_resultats_proba[a][89],aray_resultats_proba[a][90],aray_resultats_proba[a][91],aray_resultats_proba[a][92],aray_resultats_proba[a][93],aray_resultats_proba[a][94],aray_resultats_proba[a][95],aray_resultats_proba[a][96],aray_resultats_proba[a][97],aray_resultats_proba[a][98]]
                spamwriter.writerow(proba) 
    
    return precision_test # On retourne la moyenne de la precision des donnees de test sur les 10 entrainements
    
    
    
    
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
                [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement)
            
                if(precision_test > maximum):
                    maximum = precision_test
                    meilleur_premier_parametre=n_estimators
                    meilleur_second_parametre=learning_rate
      
    elif(numero_algorithme==1): # 1 = Random-Forest Classifier
        for n_estimators in liste_parametres[0]:
            for max_depth in liste_parametres[1]:
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
                [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement)
            
                if(precision_test > maximum):
                    maximum = precision_test
                    meilleur_premier_parametre=n_estimators
                    meilleur_second_parametre=max_depth
 
    elif(numero_algorithme==2): # 2 = ADA-Boost Classifier
        for n_estimators in liste_parametres[0]:
            for learning_rate in liste_parametres[1]:
                clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
                [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement)
            
                if(precision_test > maximum):
                    maximum = precision_test
                    meilleur_premier_parametre=n_estimators
                    meilleur_second_parametre=learning_rate

    elif(numero_algorithme==3): # 3 = Decision Tree Classifier
        for min_samples_split in liste_parametres[0]:
            for max_depth in liste_parametres[1]:
                clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth, random_state=0)
                [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement)
            
                if(precision_test > maximum):
                    maximum = precision_test
                    meilleur_premier_parametre=min_samples_split
                    meilleur_second_parametre=max_depth
      
    elif(numero_algorithme==4): # 4 = SVM
        for C in liste_parametres[0]:
            for kernel in liste_parametres[1]:
                clf = svm.SVC(C=C, kernel=kernel, probability=True)
                [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement)
            
                if(precision_test > maximum):
                    maximum = precision_test
                    meilleur_premier_parametre=C
                    meilleur_second_parametre=kernel
        
    elif(numero_algorithme==5): # 5 = K-Nearest Neighbors
        for n_neighbors in liste_parametres[0]:
            for algorithm in liste_parametres[1]:
                clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
                [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement)
            
                if(precision_test > maximum):
                    maximum = precision_test
                    meilleur_premier_parametre=n_neighbors
                    meilleur_second_parametre=algorithm

    elif(numero_algorithme==6): # 6 = Linear Discriminant Analysis
        for solver in liste_parametres[0]:
            for tol in liste_parametres[1]:
                clf = LinearDiscriminantAnalysis(solver=solver, tol=tol)
                [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement)
            
                if(precision_test > maximum):
                    maximum = precision_test
                    meilleur_premier_parametre=solver
                    meilleur_second_parametre=tol

    elif(numero_algorithme==7): # 7 = Quadradic Discriminant Analysis
        for n_estimators in liste_parametres[0]:
            for max_samples in liste_parametres[1]:
                clf = BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples)
                [precision_entrainement,precision_test] = Valid_croisee(clf,x_entrainement,t_entrainement)
            
                if(precision_test > maximum):
                    maximum = precision_test
                    meilleur_premier_parametre=n_estimators
                    meilleur_second_parametre=max_samples

    return meilleur_premier_parametre,meilleur_second_parametre
    
def Valid_croisee(clf,x_entrainement,t_entrainement): 
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

    return precision_entrainement,precision_test

if __name__ == "__main__":
    main()
