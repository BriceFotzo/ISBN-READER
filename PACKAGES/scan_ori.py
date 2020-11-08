__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import cv2
import numpy as np
import argparse

from transform import Transform
from basic_image import BasicImage
from combine_images import CombineImages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

""" Arugment Parser """
ap = argparse.ArgumentParser()
ap.add_argument('-i',
    '--image',
    required = True,
    help     = 'path to the image')

ap.add_argument('-H',
    '--height',
    required = False,
    default  = 300,
    help     = 'height of the image image we will process and use for finding the contours (default: 300)')

ap.add_argument('-n',
    '--noise',
    required = False,
    default  = 0,
    help     = 'the level to which we remove noise and smaller details from the scan (default: 0, i.e. preserve everything')

ap.add_argument('-cx',
    '--closingx',
    required = False,
    default  = 3,
    help     = 'the size of the closing element after applying the Canny edge detector')

ap.add_argument('-cy',
    '--closingy',
    required = False,
    default  = 3,
    help     = 'the size of the closing element after applying the Canny edge detector')

ap.add_argument('-m',
    '--multi',
    action   = 'store_true',
    required = False,
    default  = False,
    help     = 'the size of the closing element after applying the Canny edge detector')

ap.add_argument('-a',
    '--auto',
    required = False,
    action   = 'store_true',
    default  = False,
    help     = 'if we want to have automatically set values for the height and closing when looking for objects')

ap.add_argument('-s',
    '--save',
    action   = 'store_true',
    default  = False,
    help     = 'set the flag in order to save the extracted images to the current folder')
ap.add_argument('-r',
    '--isbn_zone',
    required = False,
    default  = 'scan',
    help     = 'where to store the isbn_zone')
args         = vars(ap.parse_args())




# Getting the user input
HEIGHT              = int(args['height'])
NOISE_REMOVAL_LEVEL = max(int(args['noise']) * 2 - 1, 0)
CLOSING_SIZE_X        = int(args['closingx'])
CLOSING_SIZE_Y       = int(args['closingy'])
bi                  = BasicImage(args['image'])
multi=args['multi']
output = str(args['isbn_zone'])
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
            cv2.THRESH_BINARY,255,1)
    
    edgeds = auto_canny(gray)
    
    kernels = cv2.getStructuringElement(cv2.MORPH_RECT, (cx,cy))
    closeds = cv2.morphologyEx(edgeds, cv2.MORPH_CLOSE, kernels)
    cnts = cv2.findContours(closeds.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    all_areas=[cv2.contourArea(rect) for rect in cnts]
    total_area=np.sum(all_areas)
    return cnts,all_areas,closeds,threshes

def scan():
    """ Step 1: Edge Detection """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # get the grayscale image
    #gray = cv2.bilateralFilter(gray, 11, 20, 20)
    #gray = cv2.GaussianBlur(gray, (3, 3), 0) # with a bit of blurring
    #BasicImage(gray).show()

    # automatic Canny edge detection thredhold computation
    #cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    #thresh_im = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #low_thresh = high_thresh / 2.0
    # zero-parameter automatic Canny edge detection (method 2)
    # Vary the percentage thresholds that are determined (in practice 0.33 tends to give good approx. results)
    # A lower value of sigma  indicates a tighter threshold, whereas a larger value of sigma  gives a wider threshold.
    #sigma       = 0.33
    #v           = np.median(gray)
    #low_thresh  = int(max(0, (1.0 - sigma) * v))
    #high_thresh = int(min(255, (1.0 + sigma) * v))

    #edged = auto_canny(gray) # detect edges (outlines) of the objects
    #BasicImage(edged).show()

    # since some of the outlines are not exactly clear, we construct
    # and apply a closing kernel to close the gaps b/w white pixels
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSING_SIZE_X, CLOSING_SIZE_Y))
    #closed = cv2.morphologyEx(thresh_im, cv2.MORPH_CLOSE, kernel)
    #BasicImage(closed).show()

    """ Step 2: Finding Contours """
    
    sequential=sequence(image,CLOSING_SIZE_X, CLOSING_SIZE_Y)
    contours=sequential[0]
    closed=sequential[2]
    
    total = 0

    # looping over the contours found
    approx_all = []
    for contour in contours:
        # approximating the contour
        contour = cv2.convexHull(contour)
        peri    = cv2.arcLength(contour, True)
        approx  = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area    = cv2.contourArea(contour)

  

        # we don't consider anything less than 5% of the whole image
        if area < 0.005 * total_area:
            continue

        # if the approximated contour has 4 points, then assumer it is a book
        # a book is a rectangle and thus it has 4 vertices
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
            approx_all.append(approx)
            total += 1

    print ('Found %d books/papers in the image.' % total)
    #BasicImage(image).show()

    # no point of displaying anything if we couldn't find any books
    if total != 0:
        """ Displaying all intermediate steps into one image """
        top_row = CombineImages(300, original, gray)
        bot_row = CombineImages(300, closed, image)
        BasicImage(top_row).show()
        BasicImage(bot_row).show()
        #com_img = np.vstack((top_row, bot_row))
        #BasicImage(com_img).show()

        """ Step 3: Apply a Perspective Transform and Threshold """
        total = 0
        for approx in approx_all:
            total += 1
            warped = Transform.get_box_transform(original, approx.reshape(4, 2) * ratio)
            #BasicImage(warped).show()

            scan_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            scan_warped = cv2.medianBlur(scan_warped, NOISE_REMOVAL_LEVEL)
            scan_warped = cv2.adaptiveThreshold(scan_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
            #BasicImage(scan_warped).show()
            BasicImage(CombineImages(400, warped, scan_warped)).show()

            # save the image
            if args['save'] == True:
                filename_color = 'books/'+output+'.jpg'
                filename_scan  = 'books/scan%03d_scan.jpg' % total
                BasicImage(warped).save(filename_color)
                BasicImage(scan_warped).save(filename_scan)

    return total

if args['auto'] == False:
    original   = bi.get().copy()
    ratio      = original.shape[0] / float(HEIGHT)
    image      = bi.resize('H', HEIGHT)
    total_area = image.shape[0] * image.shape[1]

    #BasicImage(image).show()

    scan()
    print("scanned")
else:
    original   = bi.get().copy()
    HEIGHT       = original.shape[0]
    ratio        = original.shape[0] / float(HEIGHT)
    image        = bi.resize('H', HEIGHT)
    total_area   = original.shape[0] * original.shape[1]

            # print ('auto_height = ', auto_height)
            #print ('auto_closing= ', auto_closing)
    if scan() != 0:
        exit(0)