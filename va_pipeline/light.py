import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the frame
frame = cv2.imread('../samples/elephant.jpg')
frame1 = cv2.imread('../samples/goldenDark.jpeg')

cv2.imshow('Frame', frame)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the window

# Calculate the histogram
hist = cv2.calcHist([frame],[0],None,[256],[0,256])
print(hist)
hist1 = cv2.calcHist([frame1],[0],None,[256],[0,256])

# Plot the histogram
plt.figure(figsize=(12, 6))
plt.plot(hist)
plt.title('Histogram of Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')


# Plot the histogram
plt.figure(figsize=(12, 6))
plt.plot(hist1)
plt.title('Histogram of Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

print(np.median(hist))
print(np.median(hist1))

lower_threshold = 350
upper_threshold = 750
# Use mean of histogram to determine lighting condition
if np.median(hist) < lower_threshold:
    print('The image is too dark')
elif np.median(hist) > upper_threshold:
    print('The image is too bright')