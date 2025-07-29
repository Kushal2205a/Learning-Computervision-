import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Load and Prepare Images ---
image1 = cv2.imread('face.png')
training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

test_image = cv2.pyrDown(training_image)
test_image = cv2.pyrDown(test_image)
num_rows, num_cols = test_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 30, 1)
test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

# --- SIFT Feature Detection ---
sift = cv2.xfeatures2d.SIFT_create()
train_kp, train_desc = sift.detectAndCompute(training_gray, None)
test_kp, test_desc = sift.detectAndCompute(test_gray, None)

# --- FLANN Matcher with Lowe's Ratio Test ---
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(train_desc, test_desc, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Draw only good matches
result_img = cv2.drawMatches(training_image, train_kp,
                             test_image, test_kp,
                             good_matches[:50], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# --- Show Result ---
plt.figure(figsize=(24, 12))
plt.title('Top 50 Filtered Matches (Lowe\'s Ratio Test)')
plt.imshow(result_img)
plt.axis('off')
plt.tight_layout()
plt.show()

# Print stats
print("Original matches:", len(matches))
print("Good matches after Lowe's ratio test:", len(good_matches))
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Load and Prepare Images ---
image1 = cv2.imread('face.png')

# Convert to RGB
training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

# Create test image by scaling down and rotating
test_image = cv2.pyrDown(training_image)
test_image = cv2.pyrDown(test_image)
num_rows, num_cols = test_image.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 30, 1)
test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

# --- Show Original & Transformed Images ---
fx, plots = plt.subplots(1, 2, figsize=(20, 10))
plots[0].set_title("Training Image")
plots[0].imshow(training_image)
plots[1].set_title("Testing Image")
plots[1].imshow(test_image)

# --- Detect Keypoints and Descriptors using SIFT ---
sift = cv2.xfeatures2d.SIFT_create()
train_keypoints, train_descriptor = sift.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)

# Draw keypoints
keypoints_without_size = np.copy(training_image)
keypoints_with_size = np.copy(training_image)
cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color=(0, 255, 0))
cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
fx, plots = plt.subplots(1, 2, figsize=(20, 10))
plots[0].set_title("Train keypoints With Size")
plots[0].imshow(keypoints_with_size)
plots[1].set_title("Train keypoints Without Size")
plots[1].imshow(keypoints_without_size)

# Print number of keypoints
print("Number of Keypoints Detected In The Training Image:", len(train_keypoints))
print("Number of Keypoints Detected In The Query Image:", len(test_keypoints))

# --- Match Keypoints ---
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
matches = bf.match(train_descriptor, test_descriptor)

# Sort matches by distance and keep top 100
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:100]

# Draw matches on COLOR images
result = cv2.drawMatches(training_image, train_keypoints,
                         test_image, test_keypoints,
                         good_matches, None, flags=2)

# Show the matches
plt.figure(figsize=(20, 10))
plt.title('Top 100 SIFT Matches (Color)')
plt.imshow(result)
plt.axis('off')
plt.show()

# Print number of good matches
print("\nNumber of Matching Keypoints Between The Training and Query Images:", len(matches))
