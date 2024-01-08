# Monitoring Highway Construction Activity
A highway construction monitoring task for a patch of 10km of NH 275 is taken for demonstration purposes. The road construction activity is monitored by post-classification comparison and change detection techniques.

# Introduction
Construction monitoring is a process of inspection to know whether the project proposed is consistent with the plans and funds processed. The methodologies range from simple observation of construction practices, and physical measurement of size, spacing, and depths, to vertical displacement. With access to high-resolution satellite imagery and advanced data processing algorithms, construction companies can remotely monitor and manage their projects, resulting in improved project efficiency and reduced costs. Satellite images offer detailed information about the site, including land use, topography, and vegetation cover, which can assist construction companies in project planning and decision-making.

# Problem Statement
To horizontally monitor NH-275, commonly called as Bengaluru-Mysuru Expressway using open-source satellite imagery.

# Theory
## Land use/Land cover map:
Land use/Land cover (LULC) maps represent spatial information of different types/classes of physical coverage of the Earth’s surface. The generation of LULC maps establishes the baseline information for activities like change detection. LULC map is prepared by classifying the satellite images using the random forest (RF) classifier.

## Random Forest Algorithm:
Random forest is a supervised learning algorithm useful for classification by using multiple decision trees. From training sets, a random subset is taken, and individual decision trees are constructed. Each decision tree gives a vote for a class. The class with majority voting will be assigned to the pixel.

## Image Differencing:
Image differencing is band-by-band, pixel-by-pixel subtraction of two images. Each band of the subtracted image contains the differences between the spectral values of the pixels in the two original bands.

## Otsu Thresholding:
Otsu’s method is a variance-based technique to find the threshold value where the weighted variance between the foreground and background pixels is the least. The key idea here is to iterate through all the possible values of the threshold and measure the variance of background and foreground pixels. Then find the threshold where the within-class variance is the least.

# Methodology
The random Forest model is known to be the best machine-learning model to do LULC classification. Hyperparameters of the RF model are set by a few test trials and 100 decision trees to classify satellite images. The sampling dataset is created of forty polygons in each category with Sentinel-2 satellite imagery as the base map. This can be carried out in QGIS, ArcGIS, or any GIS software by annotators.
Six satellite images of AOI are classified by reusing the samples created on one satellite image. The signature file used to train the model classifies new images with similar accuracy. The images are classified into six categories: Water, agricultural lands, forest, barren, urban, and roads.
A simple change detection technique is applied to detect changes in road construction activity. The image differencing technique determines changes between two images. The brighter pixels indicate change pixels and the darker pixels indicate no-change pixels. Otsu thresholding creates a binary image showing the changes with pixel value '1' and no change as '0'. <br/>

<div align="center">
<img width="200" src="https://github.com/aishwarya-10/change_detection/assets/48954230/cc99998c-b796-4fe6-af14-e9c6622a7f46">
</div>

# Results
## Dataset details:
<div align="center">
<img width="600" src="https://github.com/aishwarya-10/change_detection/assets/48954230/6be353a9-b676-43ad-ba1b-3445120c6418">
</div>

## Accuracy:
The overall accuracy of LULC maps is found to be 97.14%.

## Discussion:
Time-series LULC maps are generated to show the class-wise changes in the construction activity. LULC maps are prepared from April 2019 to March 2023 on two images per year basis. April 2019 shows no new road construction and appears barren mostly. November 2019 appeared much greener compared to the previous image because of seasonal changes and this indicates the starting of earthwork (barren class). Further in March 2021, the class road confirms the construction has been completed in a particular section. Finally, the completed Bengaluru-Mysuru expressway can be seen in March 2023.

Image differencing determines the exact progress of an activity during a specified period. <br/>
<div align="center">
<img src="https://github.com/aishwarya-10/change_detection/assets/48954230/2064989e-e8f2-49be-b22d-50e454e83a60"> <br/>

<img src="https://github.com/aishwarya-10/change_detection/assets/48954230/42262b88-98a8-4383-9ded-926172cef817">
</div>

# Conclusion
Identifying feature-wise changes in LULC images is quite helpful in determining the type of change. Preparing a time-series LULC can be a time-consuming task. Although reusing the polygon sample signature file of one image for additional images aids in categorization, the accuracy decreases as the season progresses. As a result, a sample dataset for summer and winter season images can be generated and used for those season images.
