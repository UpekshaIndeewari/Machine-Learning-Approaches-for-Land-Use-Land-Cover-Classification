# Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification
This  project  explores the application of machine learning algorithms in identifying and mapping different land cover types using satellite  imagery and  compares the performance of  algorithms.

## Abstract
This project investigates machine learning (ML) techniques for land use and land cover (LULC) classification across the [Moehne Reservoir](https://en.wikipedia.org/wiki/M%C3%B6hne_Reservoir#:~:text=The%20M%C3%B6hne%20Reservoir%2C%20or%20Moehne,million%20cubic%20metres%20of%20water) region in Germany using Python and open-source Sentinel-2 satellite data. By employing ML algorithms, this study evaluates the performance of these methods for identifying LULC changes and selects the most effective algorithm. The flexible, scalable framework demonstrated in this project achieves good accuracy in LULC categorization, underscoring its critical role in regional and national-scale monitoring.

## Introduction

### 1.1 Project Background
LULC classification is a fundamental process in environmental science, involving the identification and mapping of various surface features such as vegetation, water bodies, urban infrastructure, and bare land. Understanding and monitoring LULC is crucial for effective environmental management, urban planning, agriculture, and biodiversity conservation. These classifications serve as the foundation for assessing changes in land cover over time, which is essential for tracking the impact of human activities and natural phenomena on ecosystems.

Traditional methods of LULC classification, including manual and rule-based approaches, often struggle to accurately capture the complexities of spectral and spatial patterns in heterogeneous landscapes. The Moehne Reservoir region exemplifies such diversity, with a mix of forests, water bodies, agricultural fields, and urban areas that challenge traditional classification models.

To address these limitations, this project employs ML algorithms, which offer the capability to model non-linear relationships and leverage high-dimensional datasets effectively. By analyzing Sentinel-2 imagery using ML techniques such as **Support Vector Machine (SVM)**, **Random Forest (RF)**, and **Decision Trees (DT)**, this project identifies the most effective method for LULC classification in the region. Additionally, it evaluates these algorithms to determine the most reliable and scalable solution for assessing land use changes.

### 1.2 Objectives
* To leverage open source Sentinel-2 satellite data for LULC analysis.
* To implement and evaluate machine learning algorithms for LULC classification.
* To assess the performance of machine learning algorithms in capturing LULC patterns.
* To identify the most effective machine learning algorithm for LULC classification.
* To propose a flexible and scalable framework for LULC classification.

## 2. Proposed workflow
The proposed work flow consisted with following steps.
![Proposed Work flow](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/workflow.png)

### 2.1	Data collection and Preparation
**Data**

Two types of data are used for this project
* Sentinel-2 satellite imagery is collected for the study area, capturing its spectral bands such as Red, Green, and Blue. These bands are essential for distinguishing different land cover types based on their spectral signatures. Saelite images are downloaded from [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
* A vector dataset in the form of shapefiles (.shp) is created to provide labeled sample points for training ML models. These points represent known land cover categories, such as 1- Water, 2- Agriculture, 3- Forest, 4- Urban, and 5- Barren.

Following shows the attribute table of the point shape file with relevent ID numbers.

![Attribute table for shape file](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/attribute%20table.PNG)

Following shows the lables sample points on the satelite image.

![Study area](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/area.PNG)

**Data Preprocessing**

Once the data is acquired, it needs to be preprocessed to ensure optimal quality and accuracy for the classification process. Preprocessing involves removing any noise or artifacts that may interfere with the analysis, such as clouds, atmospheric distortions, or inconsistencies in illumination. The selected imagery had minimal cloud cover, reducing the need for extensive atmospheric corrections. 
To ensure consistency and clarity in the analysis, the spectral bands are normalized. 

Normalization scales the band values to a range between 0 and 1, improving the stability and interpretability of the models. The normalized data is visualized as an RGB composite and individual band plots to confirm data quality and eliminate potential anomalies. This preprocessing step ensures that the input data is consistent and ready for use in machine learning-based Land Use and Land Cover (LULC) classification.

Following shows RGB bands for selected satelite image

![RGB Bands](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/normalized%20bands_code.png)

Following shows the normalized image 

![Normalized Satelite image](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/normalized%20image_code.png)

## 2.2	Feature Extraction
In this step, pixel values are extracted from the raster data at the locations of the sample points provided in the shape file. Each sample point is associated with a multi-band pixel value, representing its spectral characteristics. These pixel values, combined with their corresponding LULC labels, are compiled into a pandas DataFrame. The resulting table forms the dataset used for training and evaluating the machine learning models.

![Data Frame with LULC classes](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/dataframe.PNG)

To prepare the data for modeling, the dataset is split into training and testing subsets. The training set (70% of the data) is used to build the models, while the testing set (30% of the data) evaluates their performance. 

## 2.3 Machine Learning Model Development
This step focuses on developing and training the machine learning models. Three popular algorithms are selected: **Support Vector Machine (SVM), Random Forest (RF)**, and **Decision Tree (DT)**. These models are implemented using the training data to classify land cover types based on spectral band values. Each model is trained to capture the relationships between input features (band values) and output labels (LULC classes).

After training, the models are evaluated on the testing dataset. The evaluation involves calculating performance metrics such as accuracy, classification reports (precision, recall, F1-score), and confusion matrices. These metrics help assess the models' ability to correctly classify each land cover type.

## 2.4 Model Comparison
The results from all three models are compared to determine the most effective algorithm for LULC classification. Accuracy scores and confusion matrix percentages are used to rank the models. Heatmaps of the confusion matrices are visualized to understand each model's strengths and weaknesses in classifying different land cover types. Based on these comparisons, the best-performing model is selected for the next steps.

## 2.5 Exporting predicted maps
The predicted maps generated by each algorithm were exported as GeoTIFF images. These files are critical as they provide georeferenced spatial representations of the classified LULC categories, allowing for visualization and further analysis. These images can be loaded into Geographic Information Systems (GIS) software like QGIS or ArcGIS for further analysis, visualization, or integration with other spatial datasets.

GeoTIFF is a widely supported format in GIS software like QGIS or ArcGIS, ensuring interoperability and usability for future studies. During the export process, the spatial metadata from the original raster file, such as spatial resolution, coordinate reference system (CRS), and transformation parameters, were preserved to ensure spatial alignment with the input data. 

## 3. Results and discussion
The performance of the three classification models SVM, RF, and DT was analyzed based on their accuracy, precision, recall, F1-score, and confusion matrices. Here’s a detailed discussion:

**Support Vector Machine (SVM)**

SVM achieved an accuracy of **60.53%**, which was the lowest among the three models. The confusion matrix indicates that class 3 (Forest) and class 4 (Urban) were predicted with relatively high accuracy, but there were significant misclassifications in classes 1 (Water) and 5 (Barren), with precision and recall dropping to zero for these categories. This may be due to the sensitivity of SVM to overlapping spectral characteristics in the dataset or insufficient support vectors for underrepresented classes. The macro average F1-score of **44%** further confirms that SVM struggled to balance performance across all classes.

Following shows the confusion matrix of the SVM algorithm

![Confusion matrix for SVM](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/CM_SVM.PNG)

Following shows the predicted map generated from SVM

![Predicted map from SVM](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/results/Layout_SVM.jpg)

This image only shows four classes such as water, agriculture, urban and barren

**Random Forest (RF)**

RF showed the best performance with an accuracy of **86.84%**, demonstrating its capability to handle complex data distributions and identify patterns effectively. The confusion matrix reveals excellent precision and recall for most classes, with minor misclassifications for class 2 (Agriculture) and class 5 (Barren). The weighted F1-score of **86%** emphasizes its consistent performance across classes. This result underscores RF’s ability to generalize well due to its ensemble learning approach, which reduces overfitting and enhances robustness.

Following shows the confusion matrix of the RF algorithm

![Confusion matrix for RF](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/CM_RF.PNG)

Following shows the predicted map generated from RF

![Predicted map from SVM](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/results/Layout_RF.jpg)

**Decision Tree (DT)**

DT achieved an accuracy of **81.58%**, slightly lower than RF but significantly better than SVM. The confusion matrix indicates good classification for classes 1 (Water), 2 (Agriculture), and 3 (Forest), but noticeable confusion between class 4 (Urban) and class 5 (Barren). This is reflected in a moderate F1-score of **70%** for class 4 and **62%** for class 5. While DT performed well overall, its lower recall for some categories suggests sensitivity to noise and overfitting compared to RF.

Following shows the confusion matrix of the DT algorithm

![Confusion matrix for DT](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/images/CM_DT.PNG)

Following shows the predicted map generated from DT

![Predicted map from SVM](https://github.com/UpekshaIndeewari/Machine-Learning-Approaches-for-Land-Use-Land-Cover-Classification/blob/main/results/Layout_DT.jpg)

## 4. Conclusion
The **Random Forest classifier** outperformed SVM and Decision Tree in terms of overall accuracy and balanced performance across LULC classes, making it the most reliable model for this task. The Decision Tree also showed strong performance but was slightly less consistent in handling complex class boundaries. SVM, while robust in certain scenarios, struggled with this dataset, particularly for underrepresented or spectrally overlapping classes.

## 5. Improvements and Future Work

1. Data Augmentation: Addressing class imbalance by oversampling underrepresented categories (e.g., Water and Barren) could improve SVM's performance.  
2. Feature Engineering: Incorporating additional spectral bands (e.g., SWIR) or texture features could enhance model accuracy by providing richer information for classification.  
3. Hyperparameter Tuning:Optimizing hyperparameters for SVM, RF, and DT may lead to further improvements in classification performance.  
4. Ensemble Models: Combining multiple models in an ensemble framework could balance the strengths and weaknesses of individual classifiers.  
5. Validation:Testing on additional datasets and cross-validation can further confirm the robustness of these models.

## 6. References

https://link.springer.com/article/10.1007/s12145-023-01073-w

https://www.sciencedirect.com/science/article/pii/S1574954122004058

https://waleedgeo.medium.com/lulc-py-78cb954673d

https://towardsdatascience.com/land-cover-classification-in-satellite-imagery-using-python-ae39dbf2929


















