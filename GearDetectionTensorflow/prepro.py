import cv2
import glob
import numpy as np
import time
import os

def rescale_img(image):
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 1600.0 / image.shape[1]
    dim = (1600, int(image.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def crop_img(category):

    j = 0
    source_file = "raw_data/"+ category + "/*.bmp"
    destination_file = "prepro_data/"+category+"/"+category

    for img in glob.glob(source_file):
        orig_img = cv2.imread(img)
        start_time = time.time()
        im = rescale_img(orig_img)
        orig_img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        #cv2.imshow("scale imgae", im)
        #cv2.waitKey(0)

        (thresh, im_bw) = cv2.threshold(orig_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imshow("binary image", im_bw)
        # cv2.waitKey(0)
        kernel = np.ones((4, 4), np.uint8)
        closing = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)

        (_, cnts, _) = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i = 0

        list_coordx = []
        list_coordy = []
        list_coordw = []
        list_coba = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if x > 450 and x < 1100:
                if (w > 120 and h > 135) and (w < 160 and h < 200):
                    if w / h >= 0.65 and w / h <= 1:
                        roi = im[y:y + h, x:x + w]
                        list_coordx.append(x)
                        list_coordy.append(y)
                        list_coordw.append(w)
                        list_coba.append((x, y, h))
                        i = i + 1

        if (i != 4):
            print("Error..............................................................")
        height, width, channels = im.shape
        test = sorted(list_coba)
        xmax = max(list_coordx)
        xmin = min(list_coordx)
        ymax = max(list_coordy)
        ymin = min(list_coordy)
        wmax = max(list_coordw)

        if test[0][1] < test[1][1]:
            hmax = test[1][2]
        else:
            hmax = test[0][2]

        a = ymax + hmax - ymin + 40

        b = int((a - xmax - wmax + xmin) / 2)
        c = xmax + wmax + 2 * b - xmin
        if (a != c):
            new_cropped = im[(ymin - 20):(ymax + hmax) + 19, (xmin - b):(xmax + wmax) + b]
        else:
            new_cropped = im[(ymin - 20):(ymax + hmax) + 20, (xmin - b):(xmax + wmax) + b]

        tinggi, lebar, channels = new_cropped.shape
        end_time_str = " %s s" % round((time.time() - start_time), 3)

        cv2.imwrite(destination_file +"-"+ str(j) + ".bmp", new_cropped)
        print("saving to", destination_file + "-" + str(j) + ".bmp")
        print("completed created in ", end_time_str)
        j = j + 1

    print("saving data "+ category + " completed")
