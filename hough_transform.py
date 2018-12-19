#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:45:54 2018

@author: varshath
"""

import cv2
import numpy as np
import math



def flipMatrix(mat):
    n=len(mat)-1
    for i in range(int(len(mat)/2)):
        temp=mat[i]
        mat[i]=mat[n-i]
        mat[n-i]=temp
    
    n=len(mat[0])-1
    for j in range(len(mat)):
        temp=mat[j][0]
        mat[j][0]=mat[j][n]
        mat[j][n]=temp
    return mat

def sobel(img):

    #Reading the input image. Please change the image  name here if any other images is to be read
    #img = cv2.imread("task1.png",0)
    height=img.shape[0]
    width=img.shape[1]
    #Sobel operator for detecting edges along x axis is defined
    sobel=[[0 for i in range(3)] for j in range(3)]
    sobel[0][0]=-1
    sobel[1][0]=-2
    sobel[2][0]=-1
    sobel[0][2]=1
    sobel[1][2]=2
    sobel[2][2]=1
    sobel=flipMatrix(sobel)
    img_x=[[0 for j in range(width+2)] for i in range(height+2)]
    
    #padding is done
    padded_img=[[0 for j in range(width+2)] for i in range(height+2)]
    for i in range(1,height+1):
        for j in range(1,width+1):
            padded_img[i][j]=img[i-1][j-1]
            
    #Applying Sobel Operator on Image
    for x in range(1,height+1):
        for y in range(1,width+1):
            topLeft=padded_img[x-1][y-1]*sobel[0][0]
            topRight=padded_img[x-1][y+1]*sobel[0][2]
            bottomLeft=padded_img[x+1][y-1]*sobel[2][0]
            bottomRight=padded_img[x+1][y+1]*sobel[2][2]
            middleLeft=padded_img[x][y-1]*sobel[1][0]
            middleRight=padded_img[x][y+1]*sobel[1][2]
            val=topLeft+topRight+bottomLeft+bottomRight+middleLeft+middleRight
            img_x[x][y]=val
    
    maxElement=abs(img_x[0][0])
    for i in range(1,height+1):
        for j in range(1,width+1):
            if abs(img_x[i][j])>maxElement:
                maxElement=abs(img_x[i][j])
                
    for i in range(1,height+1):
        for j in range(1,width+1):
            img_x[i][j]=abs(img_x[i][j])/maxElement
    
    print("hi")
    
    #Changing Sobel Operator for detecting edges along y axis
    sobel[0][0]=-1
    sobel[0][1]=-2
    sobel[0][2]=-1
    sobel[2][0]=1
    sobel[2][1]=2
    sobel[2][2]=1
    sobel=flipMatrix(sobel)
    img_y=[[0 for j in range(width+2)] for i in range(height+2)]
    for x in range(1,height+1):
        for y in range(1,width+1):
            topLeft=padded_img[x-1][y-1]*sobel[0][0]
            topRight=padded_img[x-1][y+1]*sobel[0][2]
            bottomLeft=padded_img[x+1][y-1]*sobel[2][0]
            bottomRight=padded_img[x+1][y+1]*sobel[2][2]
            middleLeft=padded_img[x][y-1]*sobel[1][0]
            middleRight=padded_img[x][y+1]*sobel[1][2]
            val=topLeft+topRight+bottomLeft+bottomRight+middleLeft+middleRight
            img_y[x][y]=val
    maxElement=abs(img_y[0][0])
    for i in range(1,height+1):
        for j in range(1,width+1):
            if abs(img_y[i][j])>maxElement:
                maxElement=abs(img_y[i][j])
                
    for i in range(1,height+1):
        for j in range(1,width+1):
            img_y[i][j]=abs(img_y[i][j])/maxElement
    
    
    both_edges=[[0 for j in range(width+2)] for i in range(height+2)]
    
    #Calculating magnitude of gradient
    for i in range(0,height+2):
        for j in range(0,width+2):
            both_edges[i][j]=math.sqrt(math.pow(img_x[i][j],2)+math.pow(img_y[i][j],2))
    maxElement=abs(both_edges[0][0])
    for i in range(1,height+1):
        for j in range(1,width+1):
            if abs(both_edges[i][j])>maxElement:
                maxElement=abs(both_edges[i][j])
                
    for i in range(1,height+1):
        for j in range(1,width+1):
            both_edges[i][j]=abs(both_edges[i][j])/maxElement
    both_edges=np.asarray(both_edges)
    
    cv2.imwrite('sobel.jpg',both_edges*255)
    return both_edges
    
    
def threshold(img,threshold):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]>=threshold:
                img[i][j]=255
            else:
                img[i][j]=0
    return img
 

def draw_lines_on_image(image,start,end,number_of_lines,image_name,color):
    thetas = np.deg2rad(np.arange(start,end,1))
    d_range=np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    ranges =R =np.arange(-np.ceil(d_range),np.ceil(d_range),1)
    theta = np.deg2rad(np.arange(start,end, 1))
    accumulator_matrix = np.zeros(((ranges.shape[0]), (theta.shape[0])))
    y, x = np.nonzero(image)
    for i in range(y.shape[0]):
        xi=x[i]
        yi=y[i]
        for j in range(theta.shape[0]):
            d = int((xi * np.cos(theta[j]) + yi * np.sin(theta[j])) + (np.ceil(np.sqrt(image.shape[0]**2 + image.shape[1]**2))))
            accumulator_matrix[d, j] += 1
    indices=findMaxHough(accumulator_matrix,number_of_lines)
    color_img=cv2.imread("original_imgs/hough.jpg")
    for i in range(len(indices)):
        r = R[indices[i][0]]
        theta = thetas[indices[i][1]]
        x0,y0 = np.cos(theta)*r, np.sin(theta)*r
        x1 = int(x0 + 1000*(-np.sin(theta)))
        y1 = int(y0 + 1000*(np.cos(theta)))
        x2 = int(x0 - 1000*(-np.sin(theta)))
        y2 = int(y0 - 1000*(np.cos(theta)))
        if color=='g':
            cv2.line(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif color =='r':
            cv2.line(color_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imwrite(image_name,color_img)








    
def findMaxHough(accumulator_matrix,num):
    houghPoints = []
    accumulator_matrix_copy= np.copy(accumulator_matrix)
    neighbours=31/2
    for i in range(num):
        point = np.unravel_index(np.argmax(accumulator_matrix_copy), accumulator_matrix_copy.shape)
        houghPoints.append(point)
        
        if (point[0] - neighbours) < 0: 
            min_y = 0
        else: 
            min_y = point[0] - neighbours
        if ((point[0] + neighbours + 1) > accumulator_matrix.shape[0]): 
            max_y = accumulator_matrix.shape[0]
        else: 
            max_y = point[0] + neighbours + 1
            
        if (point[1] - neighbours) < 0: 
            min_x = 0
        else: 
            min_x = point[1] - neighbours
        if ((point[1] + neighbours + 1) > accumulator_matrix.shape[1]): 
            max_x = accumulator_matrix.shape[1]
        else:
            max_x = point[1] + neighbours + 1


        min_x = int(min_x)
        max_x = int (max_x)
        min_y = int(min_y)
        max_y = int(max_y)
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                accumulator_matrix_copy[y, x] = 0                
    return houghPoints

def draw_coins_on_image(image,start,end,name_to_save,r,num_of_circles):
    theta = np.deg2rad(np.arange(0,360, 1))
    cos_thetas=np.cos(theta)
    sin_thetas=np.sin(theta)
    accumulator_matrix = np.zeros(((image.shape[0]*2), (image.shape[1]*2)))
    y_a, x_a = np.nonzero(image)
    for i in range(y_a.shape[0]):
        xi=x_a[i]
        yi=y_a[i]
        for j in range(theta.shape[0]):
            x=int(xi-r*cos_thetas[j])
            y=int(yi-r*sin_thetas[j])
            #if x>=0 and x<image.shape[0] and y>=0 and y<image.shape[1]:
            accumulator_matrix[x, y] += 1
    indices=findMaxHough(accumulator_matrix,num_of_circles)
    color_img=cv2.imread("original_imgs/hough.jpg")
    for a,b in indices:
        cv2.circle(color_img,(a,b),22,(0,255,0),3)
    cv2.imwrite(name_to_save,color_img)


img_matrix = cv2.imread("original_imgs/hough.jpg",0)
image=sobel(img_matrix)
width=image.shape[1]
height=image.shape[0]


image=threshold(image,0.05)

draw_lines_on_image(image,-37,-35,8,'blue_lines.jpg','g')
draw_lines_on_image(image,-7,0,6,'red_lines.jpg','g')


radius=22
draw_coins_on_image(image,0,360,"coin.jpg",radius,17)



