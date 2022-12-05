# -*- coding: utf-8 -*-
"""


@author: Bhumika Ravichander CSUID 2791394 Project - 2
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np

############   Task 1 ###########################
def viewAllPoints():
    
    img = cv2.imread("demo.png")
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.goodFeaturesToTrack(grayScale,0, 0.05, 15)
    edges = np.int0(edges)
    for j in edges: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (0,255,0), -1) 
      
    plt.imshow(img)
    cv2.imshow('demo',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # read the image 


############   Task 2 ###########################

def view100Points():
    
    img = cv2.imread("demo.png")
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.goodFeaturesToTrack(grayScale,100, 0.05, 15)
    edges = np.int0(edges)
    for j in edges: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (0,255,0), -1) 
      
    plt.imshow(img)
    cv2.imshow('demo',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


############   Task 3 ###########################

def clusterThree():          #calculation of K means with 3 clusters    
    img = cv2.imread("demo.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.goodFeaturesToTrack(gray,100, 0.05, 15)
    count = len(edges)   #number of points
    
    arrDist1= []
    arrDist2= []
    arrDist3= []
    arrLabel= []
    
    labelOnePoints=[]
    labelTwoPoints=[]
    labelThreePoints=[]
    
    
    c1 = edges[50]
    c2 = edges[70]
    c3 = edges[90]
   
    
    while (True):
        arrDist1= []
        arrDist2= []
        arrDist3= []
        arrLabel= []
    
        labelOnePoints=[]
        labelTwoPoints=[]
        labelThreePoints=[]
       
    
        for j in range(count):    
            
            arrDist1.append(np.linalg.norm(edges[j]-c1))
            arrDist2.append(np.linalg.norm(edges[j]-c2))
            arrDist3.append(np.linalg.norm(edges[j]-c3))
            if arrDist1[j]<arrDist2[j] and arrDist1[j]<arrDist3[j]:
                arrLabel.append(1)
                labelOnePoints.append(edges[j])
            elif arrDist2[j]<arrDist1[j] and arrDist2[j]<arrDist3[j]:
                arrLabel.append(2) 
                labelTwoPoints.append(edges[j])
            elif arrDist3[j]<arrDist1[j] and arrDist3[j]<arrDist2[j]:
                arrLabel.append(3) 
                labelThreePoints.append(edges[j])
            else:print("Error")
             
        
   
        
    
    
        print("-----------------------------------------------------------------")
        newc1= np.mean(labelOnePoints,axis=0)
        newc2= np.mean(labelTwoPoints,axis=0)
        newc3= np.mean(labelThreePoints,axis=0)
        if(newc1==c1).all() and (newc2==c2).all() and (newc3==c3).all():
            print("----------Following are new centroids----------- ")
            print("-----------------------------------------------------------------")
            print(newc1,newc2,newc3)
            print("-----------------------------------------------------------------")
       
            break
        else:
            c1=newc1
            c2=newc2
            c3=newc3
        print(c1,c2,c3)
        
            
############   Task 4 & 5 ###########################    
    for j in labelOnePoints: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (0,255,0), -1)
        a,b,c,d = cv2.boundingRect(np.float32(labelOnePoints))
        cv2.rectangle(img,(a,b),(a+c,b+d),(255,0,0),3)
    
    for j in labelTwoPoints: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (255,0,0), -1) 
        a,b,c,d = cv2.boundingRect(np.float32(labelTwoPoints))
        cv2.rectangle(img,(a,b),(a+c,b+d),(0,0,255),3)
    
    for j in labelThreePoints: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (0,0,255), -1) 
        a,b,c,d = cv2.boundingRect(np.float32(labelThreePoints))
        cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,0),3)
    
    
    plt.imshow(img)
    cv2.imshow('demo',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



########################     Same steps like for K=3 in above     #########################
    
def clusterFour():           #calculation of K means with 4 clusters
    
    img = cv2.imread("demo.png")
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.goodFeaturesToTrack(grayScale,100, 0.05, 15)
    count = len(edges)   #number of points
    
    c1 = edges[50]
    c2 = edges[60]
    c3 = edges[70]
    c4 = edges[80]
    
    while (True):
        arrDist1= []
        arrDist2= []
        arrDist3= []
        arrDist4= []
        arrLabel= []
    
        labelOnePoints=[]
        labelTwoPoints=[]
        labelThreePoints=[]
        labelFourPoints=[]
       
    
        for j in range(count):    
            
            arrDist1.append(np.linalg.norm(edges[j]-c1))
            arrDist2.append(np.linalg.norm(edges[j]-c2))
            arrDist3.append(np.linalg.norm(edges[j]-c3))
            arrDist4.append(np.linalg.norm(edges[j]-c4))
            if arrDist1[j]<arrDist2[j] and arrDist1[j]<arrDist3[j] and arrDist1[j]<arrDist4[j]:
                arrLabel.append(1)
                labelOnePoints.append(edges[j])
            elif arrDist2[j]<arrDist1[j] and arrDist2[j]<arrDist3[j] and arrDist2[j]<arrDist4[j]:
                arrLabel.append(2) 
                labelTwoPoints.append(edges[j])
            elif arrDist3[j]<arrDist1[j] and arrDist3[j]<arrDist2[j] and arrDist3[j]<arrDist4[j]:
                arrLabel.append(3) 
                labelThreePoints.append(edges[j])
                
            elif arrDist4[j]<arrDist1[j] and arrDist4[j]<arrDist2[j] and arrDist4[j]<arrDist3[j]:
                arrLabel.append(4) 
                labelFourPoints.append(edges[j])
            else:print("Error")
             
        
     
        
    
    
        print("-----------------------------------------------------------------")
        newc1= np.mean(labelOnePoints,axis=0)
        newc2= np.mean(labelTwoPoints,axis=0)
        newc3= np.mean(labelThreePoints,axis=0)
        newc4= np.mean(labelFourPoints,axis=0)
        if(newc1==c1).all() and (newc2==c2).all() and (newc3==c3).all() and (newc4==c4).all():
            print("----------Following are new centroids----------- ")
            print("-----------------------------------------------------------------")
            print(newc1,newc2,newc3,newc4)
            print("-----------------------------------------------------------------")
       
            break
        else:
            c1=newc1
            c2=newc2
            c3=newc3
            c4=newc4
        print(c1,c2,c3,c4)
        
            
    
    for j in labelOnePoints: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (0,255,0), -1)
        # a,b,c,d = cv2.boundingRect(np.float32(labelOnePoints))
        # cv2.rectangle(img,(a,b),(a+c,b+d),(255,0,0),3)
    
    for j in labelTwoPoints: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (255,0,0), -1) 
        # a,b,c,d = cv2.boundingRect(np.float32(labelTwoPoints))
        # cv2.rectangle(img,(a,b),(a+c,b+d),(0,0,255),3)
    
    for j in labelThreePoints: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (0,0,255), -1) 
        # a,b,c,d = cv2.boundingRect(np.float32(labelThreePoints))
        # cv2.rectangle(img,(a,b),(a+c,b+d),(255,255,0),3)
    
    
    for j in labelFourPoints: 
        
        x, y = j.ravel() 
        cv2.circle(img, (x, y), 3, (255,255,0), -1) 
        # a,b,c,d = cv2.boundingRect(np.float32(labelFourPoints))
        # cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,0),3)
    
    
    plt.imshow(img)
    cv2.imshow('demo',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print('''
    Please select any one option.Press 1 or 2 or 3 or 4 only

    [1] - for three clusters where K=3 with bounding Box
    [2] - for four clusters where K=4 with bounding Box
    [3] - to view all detected points on the image
    [4] - to view 100 Strongest detected points on the image
      ''')

selection=input('Please select an option (1 or 2 or 3 or 4)  ')


if selection =='1':
    clusterThree();
if selection =='2':
    clusterFour();   
if selection =='3':
    viewAllPoints();
if selection =='4':
    view100Points()








    
    
    
    
    
