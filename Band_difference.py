import rasterio as rio
from rasterio.plot import show # Visualization
from rasterio import windows
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms # histogram matching
import cv2 # Manual thresholding

path = "C://Users//Aishwarya//OneDrive - AZISTA INDUSTRIES PVT LTD//VS CODE//Change_detection//"

data1 = "S2_BMH//S2_SR_BMH_04_2019.tif"
data2 = "S2_BMH//S2_SR_BMH_11_2019.tif"
data3 = "S2_BMH//S2_SR_BMH_04_2020.tif"
data4 = "S2_BMH//S2_SR_BMH_12_2020.tif"
data5 = "S2_BMH//S2_SR_BMH_03_2021.tif"
data6 = "S2_BMH//S2_SR_BMH_12_2021.tif"
data7 = "S2_BMH//S2_SR_BMH_03_2022.tif"
data8 = "S2_BMH//S2_SR_BMH_12_2022.tif"
data9 = "S2_BMH//S2_SR_BMH_03_2023.tif"

# Read RED band
im1 = rio.open(data1).read(4)
im2 = rio.open(data9).read(4)

# Shape of image
row = im1.shape[0]
column = im1.shape[1]

print(im2.shape)
print(im2.dtype)

# plt.figure()
# plt.title("Original image")
# plt.imshow(im2)
# plt.show()

# # RGB Visualization
# RGB_im = cv2.normalize(im1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# show(RGB_im)


#-------------------------Pre-process the data--------------------#
# Image coordinates of each block
# coords = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
def coords_list(row, column, xBlockSize, yBlockSize):
  """
  ARGS: row, column, xBlockSize, yBlockSize
  RETURN: Image coordinate list
  """
  coords = []
  # loop through columns
  for j in range(0, column, xBlockSize):
    if j + xBlockSize < column:
      numCols = xBlockSize
    else:
      numCols = column - j

    for i in range(0, row, yBlockSize):
      if i + yBlockSize < row:
        numRows = yBlockSize
      else:
        numRows = row - i

      # read data
      coords.append((j, i, j + numCols, i + numRows))
  return coords


# Histogram Equalization
def npHistEq(img):
  """
  Global Histogram Equalization
    ARGS: Input image
    RETURN: Histogram equalized image
  """
  hist, bins = np.histogram(img.flatten(), 65536, [0,65536]) # Compute the cumulative distribution function (CDF) of the image
  cdf = hist.cumsum()
  cdf_normalized = cdf * hist.max() / cdf.max() # Apply the CDF to the image using linear interpolation
  equalized_img = np.interp(img.flatten(), bins[:-1], cdf_normalized) # Reshape the output image and convert it back to 16-bit
  equalized_img = equalized_img.reshape(img.shape).astype('uint16')
  return equalized_img

def npHist_eq(pre_image, post_image, coords_list):
  """
  Local Histogram Equalization
    ARGS: Pre_image, Post_image, Image_coordinate_list
    RETURNS: histogram equalized pre image and post image
  """
  arr_pre1 = pre_image.copy()
  arr_post1 = post_image.copy()

  for k in coords_list:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = k
    image_pre = pre_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] # pre-image 
    image_post = post_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] # post-image 
    # Convolution
    pre_hist_eq = npHistEq(image_pre)
    post_hist_eq = npHistEq(image_post)
    # Write the block to exact image coordinates
    arr_pre1[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = pre_hist_eq
    arr_post1[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = post_hist_eq
  return arr_pre1, arr_post1


# Histogram-matching
def hist_match(pre_image, post_image, coords_list):
  """
  ARGS: Pre_image, Post_image, Image_coordinate_list
  RETURN: histogram matched post image
  """
  arr_pre1 = pre_image.copy()
  arr_post1 = post_image.copy()

  for k in coords_list:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = k
    ref_image = pre_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] # pre-image = reference image
    true_image = post_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] # post-image = True image
    # Convolution
    matched_Im = match_histograms(true_image, ref_image, channel_axis = None) # Image, Reference, Channel
    # Write the block to exact image coordinates
    arr_post1[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = matched_Im
  return arr_post1

# # Define xBlockSize and yBlockSize for histogram equalization and normalization
# xBlockSize = 512
# yBlockSize = 512

# # Generate patch coordinates
# coords_hist = coords_list(row, column, xBlockSize, yBlockSize)

# # Histogram equalization of pre and post image
# arr_pre_eq, arr_post_eq = npHist_eq(im1, im2, coords_hist)
# # arr_pre_eq = arr_pre
# # Match histogram of post image
# arr_post_match = hist_match(arr_pre_eq, arr_post_eq, coords_hist)


#-------------------change detection--------------------#
# Thresholding
"""
  ARGS:  grayLevelImage, thresholdValue, maxValue, thresholdType
         Threshold type: (1) cv2.THRESH_BINARY (2) cv2.THRESH_BINARY_INV (3) cv2.THRESH_OTSU......
  RETURNS: Return value, Binary image
"""

# Positive change
# binary = arr_post_match - arr_pre_eq
binary = im2 - im1
maxVal = binary.max()
ret, bw_img = cv2.threshold(binary, 200, maxVal, cv2.THRESH_OTSU) 

# Negative change
# binary2 = arr_pre_eq - arr_post_match 
binary2 = im1 - im2
maxVal2 = binary2.max()
ret2, bw_img2 = cv2.threshold(binary2, 200, maxVal2, cv2.THRESH_OTSU)


# Visualization
plt.figure(figsize=(20,40))
plt.subplot(2,2,1)
plt.imshow(bw_img, cmap='gray')
plt.title("Positive Change - threshold")

plt.subplot(2,2,2)
plt.imshow(bw_img2, cmap='gray')
plt.title("Negative Change - threshold")

plt.subplot(2,2,3)
plt.imshow(im1, cmap='gray')
plt.title("im1")

plt.subplot(2,2,4)
plt.imshow(im2, cmap='gray')
plt.title("im2")

plt.show()

# Write
"""
src = rio.open(data8)
kwargs = src.meta
kwargs.update(driver = 'GTiff')

out_path = "C://Users//Aishwarya//OneDrive - AZISTA INDUSTRIES PVT LTD//VS CODE//Change_detection//highway_results"
with rio.open(out_path+'//change_2019_2023.tif', 'w', **kwargs) as dst:
        dst.write_band(1, bw_img.astype(rio.float32))

src = None
dst = None
# """
