PROJECT_DIR="C:/Users/brice/Data_science_project"
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
from matplotlib import pyplot as plt
import numpy as np
import os,sys
import pandas as pd
import shutil
from PACKAGES.process import *
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import collections
data_dir=PROJECT_DIR+'/DATA/'
training_dir=PROJECT_DIR+'/DATA/training_data/'

#importation de l'image du livre
im_to_split=cv2.imread(PROJECT_DIR+'/scan001_color.jpg')

#réduire la taille du livre(pour visualisation avec cv2.imshwo())
im_r_to_split=resize(im_to_split,'down',8,8)[0]
#rogner le livre(on ne récupère que la partie inférieure du livre)
im_bottom_ori=im_r_to_split[7*int(im_r_to_split.shape[0]/10):,:]
gray = cv2.cvtColor(im_bottom_ori, cv2.COLOR_BGR2GRAY)
gray=cv2.fastNlMeansDenoising(gray,None,10,7,21)
#application d'un filtre noir et blanc (Cette fonction affecte une valeur
#aux pixel supérieur à une valeur seuille et 0 aux autres)
threshes = cv2.adaptiveThreshold(gray,                         # image d'entrée
                              255,                            # valeur affectée aux pixels supérieurs aux  seuils
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # méthode fe filtrage à utiliser
                              cv2.THRESH_BINARY_INV,# BINARY_INV permet d'avoir le fond en noir et et les objets en blanc
                              11,       # taille de voisinage une utilisée pour calculer les valeurs seuils
                              3)
#détection des contours
edgeds = auto_canny(gray)
#définition d'un noyau de filtrage
kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
#remplissage des espaces vides pour relier des objets proches

closeds = cv2.morphologyEx(edgeds, cv2.MORPH_CLOSE, kernels)
closeds = cv2.dilate(closeds, None, iterations = 3)
cv2.imshow('Basic Image',closeds)
cv2.waitKey(0)
#augmenter la taille livre(pour visualisation avec cv2.imshow())
im_bottom_r,Rx,Ry=resize(im_bottom_ori.copy(),'up',3,5)
#identification et tracé de contours autour du code bar
im_bottom_r=draw_countours(im_bottom_r,im_bottom_ori,'code_0',15,15,Rx,Ry)

digit_dir = data_dir+'/digits'
try:
     shutil.rmtree(digit_dir)
except OSError as e:
    print("Error: %s : %s" % (digit_dir, e.strerror))

top=cv2.imread(data_dir+"result/code_0.jpg")
#agrandissement de l'image pour une meilleure détection
top=cv2.resize(top,(top.shape[1]*4,top.shape[0]*2))
# retrait des couleurs de l'image
gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
#binarisation de l'image
threshes =  cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY_INV,15,5)
#tracé des contours
edgeds = auto_canny(threshes)
#détermination ndiu noyau de filtrage(pour le remplissage)
kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (3,2))
#remplissage des vide pour la liaison des pixels formants un même chiffre
closeds = cv2.morphologyEx(threshes, cv2.MORPH_CLOSE, kernels,iterations=1)

#récupération des contours des objets trouvés
cnts = cv2.findContours(closeds.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#récupération des aires des objets trouvés
all_areas=[cv2.contourArea(rect) for rect in cnts]
#Si on a moins de 13 objets alors on a pas nos 13 digits
#Dans ce cas, on agrandit l'objet et afin de mieux identifier les objets
if len(all_areas)<15:
    gray=resize(gray,'up',2,3)[0]
    top_,Rx,Ry=resize(top.copy(),'up',2,3)
    threshes =  cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY_INV,255,30)
    edgeds = auto_canny(threshes)
    threshes= cv2.dilate(threshes, None, iterations = 1)
    kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (4,7))
    closeds = cv2.morphologyEx(threshes, cv2.MORPH_CROSS, kernels,iterations=2)
    cnts = cv2.findContours(closeds.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    all_areas=[cv2.contourArea(rect) for rect in cnts]
    min_area,max_area=100,5000
    min_p,max_p=0.1,2

else :
    top_,Rx,Ry=top.copy(),1,1
    min_area,max_area=10,8000
    min_p,max_p=0.1,2
areas_digits=pd.Series([],dtype='float64')
contours_bon=[]
#tris des countours correspondants à des chiffres de code barre (exemple: rejet d'un point/bruit apparaissant comme objet)
for i,cnt in enumerate(cnts):
    #récupération des positions x,y, tailles et hauteurs
    x,y,w,h = cv2.boundingRect(cnt)
    if (w/h <= max_p) & (w/h >= min_p) & (h/top.shape[0] < 0.3) & (y > 2*h/3) & (w*h > min_area) & (w*h < max_area) :
      contours_bon.append(cnt)

#récupération des 13 chiffres du code bar et sauvegarde daans une série de fichiers
for i,cnt in enumerate(contours_bon[:13]):
    x,y,w,h = cv2.boundingRect(cnt)
    x_,y_,w_,h_=int(x/Rx),int(y/Ry),int(w/Rx),int(h/Ry)
    #tracé des contours autour des digits sur la copie car nous voulons enregistrer l'image originale sans trait
    cv2.rectangle(top_,(x,y),(x+w,y+h),(0,0,255),1)
    #les fichiers sont nommés grâce à la position de chaque digit dur l'image
    cv2.imwrite(data_dir+"digits/"+str(x)+".jpg",top[y_-3:y_+h_+3,x_-3:x_+w_+3])

#importation des données
images=np.loadtxt(training_dir+"/flattened_images.txt", np.float32)
classes=np.loadtxt(training_dir+"/classifications.txt", np.float32)
#transformer les caractères saisis au clavier en string
classes_str=np.array([chr(int(c)) for c in classes])
#données à utiliser pour l'algorithme KNN de OpenCV
labels=classes.reshape((classes.size, 1))
#données à utiliser pour le réseau de neuronnes
targets=pd.get_dummies(pd.DataFrame(classes_str)).to_numpy()
#données à utiliser pour Sklearn
classes=[chr(int(c)) for c in classes]

print("Structure des images : {}".format(np.shape(images)))
print("Structure des lables pour Sklearn : {}".format(np.shape(classes)))
print("Structure des lables pour OpenCV: {}".format(np.shape(labels)))
print("Structure des lables pour TensorFlow : {}".format(np.shape(targets)))

ovr_clf = OneVsRestClassifier(estimator=SVC(random_state=0,C=1))
svm_clf=SVC(gamma=0.0001,C=1)
knn_clf = KNeighborsClassifier(n_neighbors=1)
kNearest_clf = cv2.ml.KNearest_create()

ovr_clf.fit(images,classes)
svm_clf.fit(images,classes)
knn_clf.fit(images,classes)
kNearest_clf.train(images, cv2.ml.ROW_SAMPLE,labels)

# on récupère les fichiers images contenant les digits
digit_files = [c for c in os.listdir(data_dir + "digits") if len(c) < 10]
# on les trie par numéro de fichier(position sur l'image code bar)
digit_files = sorted(digit_files, key=get_pos, reverse=False)
# création des listes de valeurs à récupérer
NN_List_Results, OVR_List_Results, SVM_List_Results, KNN_List_Results, KNN_Cv_List_Results, \
NN_List, OVR_List, SVM_List, KNN_List, KNN_Cv_List = [], [], [], [], [], [], [], [], [], []
for i, digit in enumerate(digit_files, 1):
    # lecture des digits
    im = cv2.imread(data_dir + "digits/" + digit)
    try:
        imgROI, w, h, imgResised = get_ROI_to_predict(im, 20, 30)
        # on modifie la taille de l'image rognée afin qu'elle soit conforme pour les modèles
        imROIResized = resize(imgROI, 't', w, h)[0]
        # on extrait les features de l'image de la zone d'intérêt
        imROIToPredict = imROIResized.reshape((1, w * h))
        # nous gardons une trace de l'image non traitée pour voir la différence si on avait extrait la zone d'intérêt avant de prédire
        im_non_processed = imgResised.reshape((1, w * h))

        # affichage des digits et des ROI représentées par des contours rectangles
        plt.subplot(1, 13, i)
        plt.imshow(imgResised, cmap='Greys')
        plt.axis('off')

        # prédiction avec et sans extraction de ROI avec KNN de OpenCV
        retval, Results, neigh_resp, dists = kNearest_clf.findNearest(np.float32(imROIToPredict), k=1)
        retval, Results_non_processed, neigh_resp, dists = kNearest_clf.findNearest(np.float32(im_non_processed), k=1)
        # récupération des résultats dans des listes
        KNN_Cv_List_Results.append(str(chr(int(Results[0][0]))))
        KNN_Cv_List.append(str(chr(int(Results_non_processed[0][0]))))

        # prédiction avec et sans extraction de ROI avec les modèles de Sklearn
        OVR_List_Results.append(str(ovr_clf.predict(imROIToPredict)[0]))
        SVM_List_Results.append(str(svm_clf.predict(imROIToPredict)[0]))
        KNN_List_Results.append(str(knn_clf.predict(imROIToPredict)[0]))
        # récupération des résultats dans des listes
        OVR_List.append(str(ovr_clf.predict(im_non_processed)[0]))
        SVM_List.append(str(svm_clf.predict(im_non_processed)[0]))
        KNN_List.append(str(knn_clf.predict(im_non_processed)[0]))

    except:
        print("Problème rencontré avec l'image")
print('Digits à prédire :')
plt.show()

print("Résultat des prédictions AVEC Extraction de ROI")
print('     Neural Network TensorFLow : |{}|'.format('|'.join(NN_List_Results)))
print('     KNN OpenCV : |{}|'.format('|'.join(KNN_Cv_List_Results)))
print('     OneVsRest Sklearn: |{}|'.format('|'.join(OVR_List_Results)))
print('     KNN Sklearn : |{}|'.format('|'.join(KNN_List_Results)))
print('     SVM Sklearn: |{}|'.format('|'.join(SVM_List_Results)))