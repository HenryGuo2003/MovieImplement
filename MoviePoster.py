import numpy as np
import Image
from glob import glob
from os.path import splitext
import cv2
from sklearn.cluster import KMeans

train_label = []
trtxtlist = open("../trainset/trainanswer.txt","r")
for trline in trtxtlist:
    gtlist = []
    intlist = []
    trline = trline.strip('\n')
    gtlist.append(trline.split(','))
    for num in gtlist:
        intlist = map(int, num)
    train_label.append(intlist)
"""
for line in Y_train:
    for x in range(3):
        try:
            print (line[x])
        except:
            break
"""

cate_0 = cate_1 = cate_2 = cate_3 = cate_4 = cate_5 = cate_6 = cate_7 = cate_8 = cate_9 = cate_10 = []
cate_11 = cate_12 = cate_13 = cate_14 = cate_15 = cate_16 = cate_17 = cate_18 = cate_19 = cate_20 = []
cate_21 = cate_22 = []

cate_dict = {0:cate_0, 1:cate_1, 2:cate_2, 3:cate_3, 4:cate_4, 5:cate_5, 6:cate_6, 7:cate_7, 8:cate_8, 9:cate_9,
        10:cate_10, 11:cate_11, 12:cate_12, 13:cate_13, 14:cate_14, 15:cate_15, 16:cate_16, 17:cate_17, 
        18:cate_18, 19:cate_19, 20:cate_20, 21:cate_21, 22:cate_22}

train_list = glob("../trainset/*.[j][p][g]")
train_list.sort()
train_count = -1
for train_jpg in train_list:  #Read training jpg
    train_count += 1
    try:
        train_image = cv2.imread(train_jpg)
    except:
        train_image = cv2.imread("../testset/999.jpg") #If error Read 999.jpg
        print ("READ ERROR!")

    train_image = cv2.resize(train_image, (200,200)) #Resize image to 200*200
    train_image = train_image[35:165, 35:165] #crop image to 130*130 from the center

    train_hsv = cv2.cvtColor(train_image, cv2.COLOR_BGR2HSV)
    train_hist = cv2.calcHist([train_hsv], [0,1], None, [50,60], [0,256, 0,256])#get HSV color histogram
    for domi in range(12):
        hist_arr = []
        hist_max = (train_hist.argmax()/len(train_hist[0]),train_hist.argmax()%len(train_hist[0]))#get max index
        train_hist[hist_max[0], hist_max[1]] = 0
        hist_arr.append(hist_max[0])
        hist_arr.append(hist_max[1])
        for cate_num in train_label[train_count]:
            cate_dict[cate_num].append(hist_arr)

#print (cate_0)
cate_0 = np.array(cate_0)
kmeans = KMeans(n_clusters=1, random_state=0).fit(cate_0)
print (kmeans.cluster_centers_)
