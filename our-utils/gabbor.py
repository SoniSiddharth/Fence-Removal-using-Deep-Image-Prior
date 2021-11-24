import numpy as np
import cv2

# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

input_image = 'simple-fence'
img = cv2.imread('./'+input_image+'.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
forw_filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

cv2.imshow('image', img)
# cv2.imshow('forward filtered image', forw_filtered_img)


back_kernel = cv2.getGaborKernel((21, 21), 8.0, 3*np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
back_filtered_img = cv2.filter2D(img, cv2.CV_8UC3, back_kernel)
# cv2.imshow('backward filtered image', back_filtered_img)

overlappedImg = cv2.addWeighted(forw_filtered_img,0.5,back_filtered_img,0.5,0)
# cv2.imshow('Blended Image',overlappedImg)

se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
bg=cv2.morphologyEx(overlappedImg, cv2.MORPH_DILATE, se)
out_gray=cv2.divide(overlappedImg, bg, scale=255)
out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 
# cv2.imwrite(input_image+"-mask.png", out_binary)
cv2.imshow('binary', out_binary)

gaussian_img = cv2.GaussianBlur(out_binary,(5,5),cv2.BORDER_DEFAULT)

kernel = np.ones((5,5), np.uint8)
img_erosion = cv2.erode(out_binary, kernel, iterations=1)
img_dilation = cv2.dilate(out_binary, kernel, iterations=1)

cv2.imshow('Input', gaussian_img)
cv2.imwrite('Erosion-fence-mask.png', img_erosion)
cv2.imshow('Erosion', img_erosion)

ret, thresh = cv2.threshold(img_erosion, 200, 255, 0, cv2.THRESH_BINARY)
# cv2.imwrite('threshold-erosion-fence-mask.png', thresh)
# cv2.imshow('threshold erosion', thresh)


from skimage import util

#cv2.drawContours(img, contours, -1, (0,255,0), 2)

thresh_eros = thresh
# thresh = cv2.bitwise_not(thresh)
# thresh_eros = cv2.cvtColor(thresh_eros, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(thresh_eros, 127, 255, 0, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Use cv2.CCOMP for two level hierarchy

# create an empty mask
mask = np.zeros(thresh_eros.shape[:2], dtype=np.uint8)
# loop through the contours
for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it
    if hierarchy[0][i][3] != -1: # basically look for holes
        # if the size of the contour is less than a threshold (noise)
        if cv2.contourArea(cnt) < 5000:
            # Fill the holes in the original image
            cv2.drawContours(thresh_eros, [cnt], 0, (255), -1)

# cv2.imshow("Img", img)
image = cv2.bitwise_not(thresh_eros, thresh_eros, mask=mask)

# cv2.imwrite(input_image+"-mask.png", image)
cv2.imshow("Mask", mask)
cv2.imwrite("TEC-"+ input_image +"-mask.png", image)
cv2.imshow("After", image)

cv2.waitKey(0)
cv2.destroyAllWindows()



