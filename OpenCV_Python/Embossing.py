import cv2
import numpy as np
img_emboss_input = cv2.imread('/home/minhhoang/Desktop/MinhHoang/OpenCV_inter/Motion-out_of_focus_Deblur/img_test/4.jpg')
cv2.imshow("ac",img_emboss_input)
cv2.waitKey()
# generating the kernels
kernel_emboss_1 = np.array([[0,-1,-1],[1,0,-1],
[1,1,0]])
kernel_emboss_2 = np.array([[-1,-1,0],
[-1,0,1],
[0,1,1]])
kernel_emboss_3 = np.array([[1,0,0],
[0,0,0],
[0,0,-1]])
# converting the image to grayscale
gray_img = cv2.cvtColor(img_emboss_input,cv2.COLOR_BGR2GRAY)
# applying #the kernels to the grayscale image and adding the offset
output_1 = cv2.filter2D(gray_img, -1, kernel_emboss_1) + 128

output_2 = cv2.filter2D(gray_img, -1, kernel_emboss_2) + 128

output_3 = cv2.filter2D(gray_img, -1, kernel_emboss_3) + 128

output_4 = output_1+output_2
cv2.imshow('Input', img_emboss_input)

#cv2.cvtColor(output_4,output_4,cv2.COLOR_BGR2GRAY)
#cv2.imshow('output_4', output_4)
cv2.imshow('Embossing - South West', output_1)
cv2.imshow('Embossing - South East', output_2)
cv2.imshow('Embossing - North West', output_3)
cv2.waitKey(0)