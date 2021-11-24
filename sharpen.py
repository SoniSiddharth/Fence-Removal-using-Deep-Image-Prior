import cv2
import numpy as np
image = cv2.imread("real-fence-man-output.png")
kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel3)

cv2.imshow('Original', image)
cv2.imshow('Sharpened', sharp_img)
    
cv2.waitKey()
cv2.imwrite('sharp-fence-man-output.png', sharp_img)
cv2.destroyAllWindows()