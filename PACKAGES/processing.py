import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
from matplotlib import pyplot as plt
import numpy as np
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged
def sequence(images,cx,cy):
    
    #gray=cv2.fastNlMeansDenoisingColored(images,None,1,8,3,3)
    #gray=cv2.fastNlMeansDenoisingColored(gray,None,3,1,3,1)

    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    gray=cv2.fastNlMeansDenoising(gray,None,10,7,21)
    threshes =  cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,3)

    edgeds = auto_canny(gray)
    
    kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (cx,cy))
    closeds = cv2.morphologyEx(edgeds, cv2.MORPH_CLOSE, kernels)
    #closeds = cv2.erode(closeds, None, iterations = 1)
    closeds = cv2.dilate(closeds, None, iterations = 3)
    cnts = cv2.findContours(closeds.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    all_areas=[cv2.contourArea(rect) for rect in cnts]
    total_area=np.sum(all_areas)
    return cnts,all_areas,closeds,threshes,images,box
#gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#gray_to_split = cv2.cvtColor(im_to_split, cv2.COLOR_BGR2GRAY)
#if im_to_split.shape[0]<im_to_split.shape[1]:
    #img=cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    #img_to_split=cv2.rotate(im_to_split, cv2.ROTATE_90_CLOCKWISE)
    #im_r=cv2.resize(img,(350,500))
    #im_r_to_split=cv2.resize(img_to_split,(350,500))
    #croped=img[1500:,:400]
#else:
    #img=im
    #img_to_split=im_to_split
    #im_r=cv2.resize(im,(350,500))
    #im_r_to_split=cv2.resize(img_to_split,(350,500))
    #croped=img[1500:,:400]

def draw_countours(image,out,cx,cy):
    sequantial=sequence(image,cx,cy)
    contours=sequantial[0]
    all_areas=sequantial[1]
    image=sequantial[4]
    box=sequantial[5]
    height=[]
    width=[]
    total_area=np.sum(all_areas)
    for cnt in contours:

        all_areas.append(cv2.contourArea(cnt))
        x,y,w,h = cv2.boundingRect(cnt)
        height.append(y+h)
        width.append(x+w)
        imc=image.copy()
        #if (y > 2*image.shape[0]/3) | (y < image.shape[0]/3) & (y+h < image.shape[0]/3):
        #bound the images
        
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        cv2.imwrite("result/"+out+".jpg",image[y:y+h,x:x+w])
        return image