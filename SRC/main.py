from matplotlib import pyplot as plt
import os
import pandas as pd
import shutil
from PACKAGES.process import *
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

PROJECT_DIR="C:/Users/brice/Data_science_project"
data_dir=PROJECT_DIR+'/DATA/'
training_dir=PROJECT_DIR+'/DATA/training_data/'

#read the back cover book's
book_cover=cv2.imread(data_dir+'books/book0.jpg')

#reduce the image size in order to show it
book_cover_reduced=resize(book_cover,'down',8,8)[0]
#split the book(we're going to keep the bottom zone of the b)
book_cover_bottom=book_cover_reduced[7*int(book_cover_reduced.shape[0]/10):,:]
cv2.imshow('Basic Image',book_cover_bottom)
cv2.waitKey(0)
# we remove the color
book_cover_gray = cv2.cvtColor(book_cover_bottom, cv2.COLOR_BGR2GRAY)
book_cover_gray=cv2.fastNlMeansDenoising(book_cover_gray,None,10,7,21)


#countours detection with the auto_canny
edgeds = auto_canny(book_cover_gray)
#defining a filtering kernel
kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))

#filling the gaps to put together the pixels that are near to each other
closed = cv2.morphologyEx(edgeds, cv2.MORPH_CLOSE, kernels)
closed = cv2.dilate(closed, None, iterations = 3)
cv2.imshow('Basic Image',closed)
cv2.waitKey(0)

#we increase the size of the image to capture the isbn number zone
book_cover_bottom_increased,Rx,Ry=resize(book_cover_bottom.copy(),'up',3,5)
#here we capture the isbn number zone and store it as a "jpg" image in a folder (isbn_zone)
isbn_zone=draw_countours(book_cover_bottom_increased,book_cover_bottom,'code_zone',15,15,Rx,Ry)

#we want to safely create a folder to store the isbn digits
digit_dir = data_dir+'/isbn_digits'
try:
     shutil.rmtree(digit_dir)
except OSError as e:
    print("Error: %s : %s" % (digit_dir, e.strerror))

isbn_zone_image=cv2.imread(data_dir+"isbn_zone/code_zone.jpg")
cv2.imshow('top',isbn_zone_image)
cv2.waitKey(0)
#we increase the isbn zone image in other to capture the digits
isbn_zone_image=cv2.resize(isbn_zone_image,(isbn_zone_image.shape[1]*4,isbn_zone_image.shape[0]*2))
isbn_zone_gray = cv2.cvtColor(isbn_zone_image, cv2.COLOR_BGR2GRAY)
#we turn the image into binary so that we can catch the digits
isbn_zone_binary =  cv2.adaptiveThreshold(isbn_zone_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY_INV,15,5)
#we draw the digits edges
isbn_zone_edged = auto_canny(isbn_zone_binary)
kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (3,2))
closed = cv2.morphologyEx(isbn_zone_binary, cv2.MORPH_CLOSE, kernels,iterations=1)

#getting the countours
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#getting the areas of all objects(we hope all digits) founded in the isbn_zone
all_areas=[cv2.contourArea(rect) for rect in cnts]
#as the isbn length is 13, we don't want to get more of 13 objects
#beside, it's possible to have have some noises
if len(all_areas)<15:
    gray=resize(isbn_zone_gray,'up',2,3)[0]
    isbn_zone_image_,Rx,Ry=resize(isbn_zone_image.copy(),'up',2,3)
    isbn_zone_binary =  cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY_INV,255,30)
    edgeds = auto_canny(isbn_zone_binary)
    isbn_zone_binary = cv2.dilate(isbn_zone_binary, None, iterations = 1)
    kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (4,7))
    closeds = cv2.morphologyEx(isbn_zone_binary, cv2.MORPH_CROSS, kernels,iterations=2)
    cnts = cv2.findContours(closeds.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    all_areas=[cv2.contourArea(rect) for rect in cnts]
    min_area,max_area=100,5000
    min_p,max_p=0.1,2

else :
    isbn_zone_image_,Rx,Ry=isbn_zone_image.copy(),1,1
    min_area,max_area=10,8000
    min_p,max_p=0.1,2
areas_digits=pd.Series([],dtype='float64')
good_contours=[]
#As we maybe caught a noise instead of digit, we want reject it
for i,cnt in enumerate(cnts):
    #we collect the positions, width and height of digits (objects caught respecting the previous rules)
    x,y,w,h = cv2.boundingRect(cnt)
    if (w/h <= max_p) & (w/h >= min_p) & (h/isbn_zone_image.shape[0] < 0.3) & (y > 2*h/3) & (w*h > min_area) & (w*h < max_area) :
      good_contours.append(cnt)

#Let's collect the digits as we just made selection
for i,cnt in enumerate(good_contours[:13]):
    x,y,w,h = cv2.boundingRect(cnt)
    x_,y_,w_,h_=int(x/Rx),int(y/Ry),int(w/Rx),int(h/Ry)
    #we draw the countours on a copy of isbnzone image
    cv2.rectangle(isbn_zone_image_,(x,y),(x+w,y+h),(0,0,255),1)
    #we save the digits naming them after theirs x positions
    cv2.imwrite(data_dir+"isbn_digits/"+str(x)+".jpg",isbn_zone_image[y_-3:y_+h_+3,x_-3:x_+w_+3])
#------------------------------------------MACHINE LEARNING-----------------------------------------#
#In this part we'll compare 4 machine learning algorithms

#loading train data (more informations about these data on the Training data folder)
images=np.loadtxt(training_dir+"/flattened_images.txt", np.float32)
classes=np.loadtxt(training_dir+"/classifications.txt", np.float32)
#turn values in strings
classes_str=np.array([chr(int(c)) for c in classes])
#labels to use with KNN algorithm of OpenCV
labels=classes.reshape((classes.size, 1))
#labels to use with Sklean algorithms
classes=[chr(int(c)) for c in classes]

print("Images Shape : {}".format(np.shape(images)))
print("Shape of labels for Sklearn : {}".format(np.shape(classes)))
print("Shape of labels for OpenCV: {}".format(np.shape(labels)))

#instanciation of models
ovr_clf = OneVsRestClassifier(estimator=SVC(random_state=0,C=1))
svm_clf=SVC(gamma=0.0001,C=1)
knn_clf = KNeighborsClassifier(n_neighbors=1)
kNearest_clf = cv2.ml.KNearest_create()

#model fitting
ovr_clf.fit(images,classes)
svm_clf.fit(images,classes)
knn_clf.fit(images,classes)
kNearest_clf.train(images, cv2.ml.ROW_SAMPLE,labels)

#with all the previous models trained, we'll predict the digits values from their respective images
# collect test data (digits stored in isbn_digits)
digit_files = [c for c in os.listdir(data_dir + "isbn_digits") if len(c) < 10]
# we sort the digits by x position to get the right order
digit_files = sorted(digit_files, key=get_pos, reverse=False)
# creating list of values to store the predictions
OVR_List_Results, SVM_List_Results, KNN_List_Results, KNN_Cv_List_Results, \
OVR_List, SVM_List, KNN_List, KNN_Cv_List = [], [], [], [], [], [], [], []
for i, digit in enumerate(digit_files, 1):
    # reading isbn digits
    im = cv2.imread(data_dir + "isbn_digits/" + digit)
    try:
        imgROI, w, h, imgResised = get_ROI_to_predict(im, 20, 30)
        # we set the size of digits images to the same values as the train data images
        imROIResized = resize(imgROI, 't', w, h)[0]
        # we extract the features from the Region Of Interest
        imROIToPredict = imROIResized.reshape((1, w * h))
        # we keep the image without the ROI exctraction so that we can compare the results
        im_non_processed = imgResised.reshape((1, w * h))

        # show the 13 digits to predict
        plt.subplot(1, 13, i)
        plt.imshow(imgResised, cmap='Greys')
        plt.axis('off')

        # predictions with KNN from OpenCV
        retval, Results, neigh_resp, dists = kNearest_clf.findNearest(np.float32(imROIToPredict), k=1)
        retval, Results_non_processed, neigh_resp, dists = kNearest_clf.findNearest(np.float32(im_non_processed), k=1)
        # we collect results
        KNN_Cv_List_Results.append(str(chr(int(Results[0][0]))))
        KNN_Cv_List.append(str(chr(int(Results_non_processed[0][0]))))

        # prediction with Sklearn models
        OVR_List_Results.append(str(ovr_clf.predict(imROIToPredict)[0]))
        SVM_List_Results.append(str(svm_clf.predict(imROIToPredict)[0]))
        KNN_List_Results.append(str(knn_clf.predict(imROIToPredict)[0]))
        # we collect results
        OVR_List.append(str(ovr_clf.predict(im_non_processed)[0]))
        SVM_List.append(str(svm_clf.predict(im_non_processed)[0]))
        KNN_List.append(str(knn_clf.predict(im_non_processed)[0]))

    except:
        print("Error related to images")
print('Digits to predict:')
plt.show()

print("Prediction results WITH ROI extraction")

print('     KNN OpenCV : |{}|'.format('|'.join(KNN_Cv_List_Results)))
print('     OneVsRest Sklearn: |{}|'.format('|'.join(OVR_List_Results)))
print('     KNN Sklearn : |{}|'.format('|'.join(KNN_List_Results)))
print('     SVM Sklearn: |{}|'.format('|'.join(SVM_List_Results)))

print("Prediction results WITHOUT ROI extraction")

print('     KNN OpenCV : |{}|'.format('|'.join(KNN_Cv_List)))
print('     OneVsRest Sklearn: |{}|'.format('|'.join(OVR_List)))
print('     KNN Sklearn : |{}|'.format('|'.join(KNN_List)))
print('     SVM Sklearn: |{}|'.format('|'.join(SVM_List)))