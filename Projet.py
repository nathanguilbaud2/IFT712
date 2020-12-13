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
    if len(sys.argv) < 2:
        usage = "\n Usage: python All_2.py choix_algorithme recherche_parametres generer_soumission\
        \n\n\t choix_algorithme:\
        \n\t\t -1 : Tous les algorithmes\
        \n\t\t 0 : Gradient Boosting Classifier\
        \n\t\t 1 : Random-Forest\
        \n\t\t 2 : ADA-Boost\
        \n\t\t 3 : Decision-Tree\
        \n\t\t 4 : SVM\
        \n\t\t 5 : K-nearest neighbors\
        \n\t\t 6 : Linear Discriminant Analysis\
        \n\t\t 7 : Quadratic Discriminant Analysis\\n"
        print(usage)
        return

    choix_algorithme = int(sys.argv[1])
    
    # On génère les données d'entraînement et de test
    generateur_donnees = gd.GestionDonnees()
    start = time.time()
    [x_entrainement,t_entrainement,x_test] = generateur_donnees.generer_donnees()
    print("------- Fin du tri des donnees en ", time.time()-start, 'seconds')
    
    # Nous allons étudier 8 algorithmes différents
    liste_variables=[[[0] for j in range(len(x_entrainement))] for j in range(7)]    
    liste_variables_test=[[[0] for j in range(len(x_test))] for j in range(7)]
    
    # Liste des parametres pour chaque algorithmes
    liste_parametres_algorithme=[ [1,1] , [100,None] , [100,1] , [6,1] , [1,"linear"] , [1,"ball_tree"] , ["svd",1] , [100,10] ]
    
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
            for variable_choisie in range(len(liste_variables)):
                Run_algorithme(liste_variables_label[variable_choisie], liste_variables[variable_choisie], t_entrainement, liste_parametres_algorithme[algorithme_choisi], algorithme_choisi)

    else: # Pour un algorithme
        print("\n",list_Algorithme_label[choix_algorithme])
        
        for variable_choisie in range(len(liste_variables)):
            Run_algorithme(liste_variables_label[variable_choisie], liste_variables[variable_choisie], t_entrainement, liste_parametres_algorithme[choix_algorithme], choix_algorithme)
    
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
 
   

if __name__ == "__main__":
    main()
