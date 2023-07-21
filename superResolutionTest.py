import cv2
import matplotlib.pyplot as plt
from Utils.superResolution import superResOpenCV

image = cv2.imread("images/family.jpg")
higRes = image.copy()
higRes = superResOpenCV(image=higRes)

cv2.imwrite('images/highRes.jpg', higRes)
cv2.imshow('Low Res', image)
cv2.imshow('High Res', higRes)

cv2.waitKey()
# fig = plt.figure(figsize=(15, 17), dpi=100)
# fig.add_subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.title('Low Res Image')


# fig.add_subplot(1, 2, 2)
# plt.imshow(higRes, cmap='gray')
# plt.axis('off')
# plt.title('High Res Image')

# plt.show()