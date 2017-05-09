import numpy as np
import Image
from glob import glob
from os.path import splitext
import cv2
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter
import math

train_label = []
trtxtlist = open("../trainset/trainanswer.txt","r") #read groundtruth
for trline in trtxtlist:
    gtlist = []
    intlist = []
    trline = trline.strip('\n')
    gtlist.append(trline.split(','))
    for num in gtlist:
        intlist = map(int, num)
    train_label.append(intlist)

cate_0, cate_1, cate_2, cate_3, cate_4, cate_5, cate_6, cate_7, cate_8, cate_9, cate_10 =([] for i in range(11))
cate_11, cate_12, cate_13, cate_14, cate_15, cate_16, cate_17, cate_18, cate_19, cate_20=([] for i in range(10))
cate_21, cate_22 = ([] for i in range(2))

center_kmeans = [] #all center of all category's kmeans
cate_temp = [] #temp save all category's index

cate_dict = {0:cate_0, 1:cate_1, 2:cate_2, 3:cate_3, 4:cate_4, 5:cate_5, 6:cate_6, 7:cate_7, 8:cate_8, 9:cate_9,
        10:cate_10, 11:cate_11, 12:cate_12, 13:cate_13, 14:cate_14, 15:cate_15, 16:cate_16, 17:cate_17, 
        18:cate_18, 19:cate_19, 20:cate_20, 21:cate_21, 22:cate_22} #dictionary of 23 category

edge_count = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] #23 category edge count
cate_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #23 category number count
face_count = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] #23 category face count
prediction_count = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]#category predict count

TP = 0.
FP = 0.
FN = 0.

train_list = glob("../trainset/*.[j][p][g]")
train_list.sort()
train_count = -1
for train_jpg in train_list:  #Read training jpg
    train_count += 1
    value_count = 0
    try:
        train_image = cv2.imread(train_jpg) #read image
    except:
        train_image = cv2.imread("../testset/999.jpg") #If error Read 999.jpg
        print ("READ ERROR!")

    train_image = cv2.resize(train_image, (200,200)) #Resize image to 200*200
    train_image = train_image[35:165, 35:165] #crop image to 130*130 from the center

    train_hsv = cv2.cvtColor(train_image, cv2.COLOR_BGR2HSV) #change image from RGB to HSV
    train_hist = cv2.calcHist([train_hsv], [0,1], None, [50,60], [0,256, 0,256])#get HS color histogram
    for domi in range(12): #get first 12 HS histogram
        hist_arr = []
        hist_max = []
        hist_max = (train_hist.argmax()/len(train_hist[0]),train_hist.argmax()%len(train_hist[0]))#get max index
        train_hist[hist_max[0], hist_max[1]] = 0 #set the max to 0
        hist_arr.append(hist_max[0])
        hist_arr.append(hist_max[1])
        for cate_num in train_label[train_count]: #put the max index in the category
            cate_dict[cate_num].append(hist_arr)

    h, s, train_value = cv2.split(train_hsv) #get picture's HSV value
    train_blurred = gaussian_filter(train_value, sigma = 3) #use gaussian filter blurred the value
    sobel_x = cv2.Sobel(train_blurred, cv2.CV_16S,1,0)#get x,y axis sobel
    sobel_y = cv2.Sobel(train_blurred, cv2.CV_16S,0,1)
    abs_x = cv2.convertScaleAbs(sobel_x)#get sobel abs of x and y 
    abs_y = cv2.convertScaleAbs(sobel_y)
    train_sobel = cv2.addWeighted(abs_x,0.5, abs_y,0.5, 0) #add to the final train sobel
    for sobel_x in range(130): #get the total edge pixel number exceed threshold 128
        for sobel_y in range(130):
            if train_sobel[sobel_x][sobel_y] >= 128:
                value_count += 1
    for cate_num in train_label[train_count]: #count every category total edge pixel number and category number
        edge_count[cate_num] = edge_count[cate_num] + value_count
        cate_count[cate_num] += 1

    train_blurred = gaussian_filter(train_image, sigma = 3)
    train_gray = cv2.cvtColor(train_blurred, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")#***Should Download first***
    train_faces = face_cascade.detectMultiScale(train_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    for cate_num in train_label[train_count]: #count every category total edge pixel number and category number
        face_count[cate_num] = face_count[cate_num] + len(train_faces)

for edge_index in range(23):#get the average of all category edge
    edge_count[edge_index] = edge_count[edge_index]/cate_count[edge_index]
    face_count[edge_index] = face_count[edge_index]/cate_count[edge_index]

for cate_num in cate_dict: #get the center of all the category's kmeans cluster
    cate_temp = []
    cate_temp = cate_dict.get(cate_num)
    cate_temp = np.array(cate_temp)
    kmeans = KMeans(n_clusters=1, random_state=0).fit(cate_temp)
    center_kmeans.append(kmeans.cluster_centers_)

print ("TRAIN kmeans done.")

test_label = []
test_txt = open("../testset/testanswer.txt","r") #read groundtruth
for test_line in test_txt:
    gtlist = []
    intlist = []
    test_line = test_line.strip('\n')
    gtlist.append(test_line.split(','))
    for num in gtlist:
        intlist = map(int, num)
        test_label.append(intlist)

test_list = glob("../testset/*.[j][p][g]")
test_list.sort()
test_count = -1
for test_jpg in test_list:
    prediction_count = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]#back to 0
    final_prediction = []
    test_count += 1
    value_count = 0
    try:
        test_image = cv2.imread(test_jpg)
    except:
        test_image = cv2.imread("../testset/999.jpg")
    test_image = cv2.resize(test_image, (200,200)) #Resize image to 200*200
    test_image = test_image[35:165, 35:165] #crop image to 130*130 from the center
    test_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV) #change image from RGB to HSV
    test_hist = cv2.calcHist([test_hsv], [0,1], None, [50,60], [0,256, 0,256])#get HS color histogram
    for domi in range(12):
        hist_arr = []
        hist_max = (test_hist.argmax()/len(test_hist[0]),test_hist.argmax()%len(test_hist[0]))#get max index
        test_hist[hist_max[0], hist_max[1]] = 0 #set the max to 0
        hist_arr.append(hist_max[0])
        hist_arr.append(hist_max[1])
        for test_index in range(23):
            prediction_count[test_index] = prediction_count[test_index] + math.sqrt(math.pow((hist_max[0]-center_kmeans[test_index][0][0]),2) + math.pow((hist_max[1]-center_kmeans[test_index][0][1]),2))

    h, s, test_value = cv2.split(train_hsv) #get picture's HSV value
    test_blurred = gaussian_filter(test_value, sigma = 3) #use gaussian filter blurred the value
    sobel_x = cv2.Sobel(test_blurred, cv2.CV_16S,1,0)#get x,y axis sobel
    sobel_y = cv2.Sobel(test_blurred, cv2.CV_16S,0,1)
    abs_x = cv2.convertScaleAbs(sobel_x)#get sobel abs of x and y 
    abs_y = cv2.convertScaleAbs(sobel_y)
    test_sobel = cv2.addWeighted(abs_x,0.5, abs_y,0.5, 0) #add to the final train sobel
    for sobel_x in range(130): #get the total edge pixel number exceed threshold 128
        for sobel_y in range(130):
            if train_sobel[sobel_x][sobel_y] >= 128:
                value_count += 1
    for test_index in range(23): #add edge value Distance
        prediction_count[test_index] = prediction_count[test_index]+abs(value_count-edge_count[test_index])

    test_blurred = gaussian_filter(test_image, sigma = 3)
    test_gray = cv2.cvtColor(test_blurred, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")#Should Download first
    test_faces = face_cascade.detectMultiScale(test_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    for test_index in range(23): #add face Distance
        prediction_count[test_index] = prediction_count[test_index]+abs(len(test_faces)-face_count[test_index])
    for domi in range(3): #get top three prediction
        min_index = prediction_count.index(min(prediction_count))
        final_prediction.append(min_index)
        prediction_count[min_index] = 10000
    temp_label = test_label[test_count]
    for test_index in temp_label: #compare prediction and groundtruth
        for prediction_index in final_prediction:
            if prediction_index == test_index:
                TP += 1.
                final_prediction[final_prediction.index(prediction_index)] = -1
                temp_label[temp_label.index(test_index)] = -1
                break

    for test_index in temp_label:
        if test_index != -1:
            FN += 1.
    for prediction_index in final_prediction:
        if prediction_index != -1:
            FP += 1.

print ("precision:")
print (TP/(TP+FP))
print ("recall:")
print (TP/(TP+FN))
print ("final acc:")
print (TP/(TP+FP+FN))
