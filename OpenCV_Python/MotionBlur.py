import cv2
import numpy as np
img = cv2.imread('//home//minhhoang//Downloads//4.jpg')
cv2.imshow('Original', img)
size = 5
# generating the kernel
kernel_motion_blur = np.zeros((size, size))
# print("zeros",kernel_motion_blur)

kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
print(kernel_motion_blur)

kernel_motion_blur = kernel_motion_blur / size
print(kernel_motion_blur)

# applying the kernel to the input image
output = cv2.filter2D(img, -1, kernel_motion_blur)
cv2.imshow('Motion Blur', output)
cv2.waitKey(0)