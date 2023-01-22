import matplotlib.pyplot as plt
import cv2
img = cv2.imread('HW2/data/2_aff.jpg') # Read image
print(img.shape)
plt.imshow(img.astype('uint8'), cmap='gray') # Show image
plt.show()