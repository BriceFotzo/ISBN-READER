import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
PROJECT_DIR="C:/Users/brice/Data_science_project"
data_dir=PROJECT_DIR+'/DATA/'
training_dir=PROJECT_DIR+'/DATA/training_data/'



def auto_canny(image, sigma=0.33):
    """
    Trace les contours sur l'image en déterminant automatiquement les valeurs min et max
    """
    # calcule la médiane des pixels contenus dans une image
    v = np.median(image)
    # détermination de valeurs minimales et maximales pour tracer les contours sur l'image
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # retourne l'image avec les contours uniquement
    return edged


def sequence(images, cx, cy):
    '''
    Applique une séquence de transformations à l'image et retourne des features
    (contours,aires des objets,image remplie,l'image en binaire)
    '''
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # application d'un filtre noir et blanc (Cette fonction affecte une valeur
    # aux pixel supérieur à une valeur seuille et 0 aux autres)
    threshes = cv2.adaptiveThreshold(gray,  # image d'entrée
                                     255,  # valeur affectée aux pixels supérieurs aux  seuils
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # méthode fe filtrage à utiliser
                                     cv2.THRESH_BINARY_INV,
                                     # BINARY_INV permet d'avoir le fond en noir et et les objets en blanc
                                     11,  # taille de voisinage utilisée pour calculer les valeurs seuils
                                     3)
    # détection des contours
    edgeds = auto_canny(gray)
    # définition d'un noyau de filtrage
    kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (cx, cy))
    # remplissage des espaces vides pour relier des objets proches
    closeds = cv2.morphologyEx(edgeds, cv2.MORPH_CLOSE, kernels)
    closeds = cv2.dilate(closeds, None, iterations=3)
    # récupérer les contours sur l'image remplie
    cnts = cv2.findContours(closeds.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # récupération des contours pour les objets dont le rapport taille_objet/taille_image \
    # est inférieur à 0.3 (règle empirique que respectent la majorité des code bar sur les livres)
    cnts_ = [cnt for cnt in cnts if cv2.boundingRect(cnt)[2] / images.shape[1] <= 0.3]
    # trie par aire pour récupérer le plus grand objet respectant la règle précédente
    c = sorted(cnts_, key=cv2.contourArea, reverse=True)[0]
    # on récupère l'inclinaison de cet objet afin l'obtenir à l'endroit après le découpage s'il est incliné
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    # récupération des aires des tous les objets trouvés
    all_areas = [cv2.contourArea(rect) for rect in cnts]
    return cnts, all_areas, closeds, threshes, images, box


def draw_countours(image, original, out, cx, cy, Rx, Ry):
    '''
    Retourne une image redimensionnée et ses nouvelles dimensions
    how détermine comment l'image est redimensionnée
    '''
    sequantial = sequence(image, cx, cy)
    contours = sequantial[0]
    all_areas = sequantial[1]
    image = sequantial[4]
    box = sequantial[5]
    x_, y_, w_, h_ = cv2.boundingRect(box)
    x, y, w, h = int(x_ / Rx), int(y_ / Ry), int(w_ / Rx), int(h_ / Ry)
    height = []
    width = []
    total_area = np.sum(all_areas)
    for cnt in contours:
        # x,y,w,h = cv2.boundingRect(cnt)
        imc = image.copy()
        cv2.drawContours(imc, [box], -1, (0, 255, 0), 3)
        cv2.imwrite(data_dir + "result/" + out + ".jpg", original[y - 3:y + h + 5, x - 15:x + w + 5])
        return imc


def resize(image, how, width, height):
    '''
    Retourne une image redimensionnée et ses nouvelles dimensions
    how détermine comment l'image est redimensionnée
    '''
    if how == 'up':
        # agrandir l'image avec des rapports width et height
        image = cv2.resize(image, (int(image.shape[1] * width), int(image.shape[0] * height)))
    elif how == 'down':
        # réduire l'image avec des rapports width et height
        image = cv2.resize(image, (int(image.shape[1] / width), int(image.shape[0] / height)))
    else:
        image = cv2.resize(image, (width, height))
    return image, width, height


def next_batch(num, data, labels):
    '''
    Retourne un total de num sets/échantillons aléatoires de données et labels
    '''
    # liste de numéros allant de 0 à la taille de data
    idx = np.arange(0, len(data))
    # réordonner la liste aléatoirement
    np.random.shuffle(idx)
    # récupérer les num premiers éléments de la liste réordonnée
    idx = idx[:num]
    # récupérer les données et labels correspondant aux num numéros récupérés
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def get_pos(char):
    '''
    Retourne la position de chaque digit
    char nom de fichier sous la forme 'position.jpg'
    '''
    return int(char.split('.')[0])


def get_ROI_to_predict(im, w, h):
    '''
    Retourne la zone d'intérêt d'une image qui va être redimensionnée (w*h) et l'image redimensionnée
    '''
    # redimensionnement(on enlève la dimension couleurs)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # on modifie la taille de l'image pour avoir celle du modèle
    im_r = resize(gray, 't', w, h)[0]
    # on floutte l'image afin d'éliminer de minimiser l'importance du bruit
    imgBlurred = cv2.GaussianBlur(im_r, (9, 9), 0)
    # on aplique un filtre à l'image pour la passer en noir et blanc
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 41, 19)
    imgThreshCopy = imgThresh.copy()
    contours, hierarchy = cv2.findContours(imgThreshCopy,
                                           # nous mettons en entrée une copie pour s'assurer que notre image d'origine ne soit pas modifiée
                                           cv2.RETR_EXTERNAL,
                                           # retrouver uniquement les contours extérieurs pour un objet
                                           cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        [intX, intY, intW, intH] = cv2.boundingRect(
            contour)  # récupérer les coordonnées du rectangle qui entoure l'objet

        # tracer ce rectangle sur l'image original
        cv2.rectangle(im_r,
                      (intX, intY),  # Côté en haut à gauche
                      (intX + intW, intY + intH),  # Côté en bas à droite
                      (0, 255, 0),  # couleur des bordures
                      1)  # épaisseur
        imgROI = imgThresh[intY:intY + intH,
                 intX:intX + intW]  # récupérer la zone/region d'intérêt (Region Of Interest)
        return imgROI, w, h, im_r