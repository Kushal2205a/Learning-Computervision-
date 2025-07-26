import cv2
import numpy as np

# Step 1: Read the image (grayscale or original)
image = cv2.imread("bike1.jpg", cv2.IMREAD_GRAYSCALE)

# Optional: Normalize to range 0-255 (if needed)
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

# Step 2: Apply color map (convert to heatmap)
heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)

# Step 3: Show or save the result
cv2.imshow("Heatmap", heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("heatmap_output.jpg", heatmap)