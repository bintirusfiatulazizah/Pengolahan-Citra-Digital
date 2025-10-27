import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Bunga.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img_rgb, (7,7), 0)

edges = cv2.Canny(gray, 100, 200)

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title('Citra Asli')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(2,2,2)
plt.title('Grayscale')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title('Gaussian Blur')
plt.imshow(blur)
plt.axis('off')

plt.subplot(2,2,4)
plt.title('Deteksi Tepi (Canny)')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()