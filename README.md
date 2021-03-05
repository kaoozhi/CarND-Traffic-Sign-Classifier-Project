# **Traffic Sign Recognition**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
## README
---
This is the repo of Traffic sign recognition project of Udacity Self-driving Car Nano degree, original repo can be found [here](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/train_data_visualization.jpg "Visualization"
[image2]: ./results/grayscale.jpg "Grayscaling"
[image3]: ./results/histo_equ.jpg "Hitrogram equalization"
[image4]: ./web_images/img01.jpg "Traffic Sign 1"
[image5]: ./web_images/img02.jpg "Traffic Sign 2"
[image6]: ./web_images/img03.jpg "Traffic Sign 3"
[image7]: ./web_images/img04.jpg "Traffic Sign 4"
[image8]: ./web_images/img05.jpg "Traffic Sign 5"
[image9]: ./web_images/img06.jpg "Traffic Sign 6"
[image10]: ./web_images/img07.jpg "Traffic Sign 7"
[image11]: ./web_images/img08.jpg "Traffic Sign 8"
[image12]: ./web_images/img09.jpg "Traffic Sign 9"
[image13]: ./web_images/img10.jpg "Traffic Sign 10"
[image15]: ./results/augmented_data_visulization.jpg "augmented data"
[image16]: ./results/original_data_sample.jpg "original data sample"
[image17]: ./results/augmented_data_sample.jpg "augmented data sample"
[image18]: ./results/model_lrn_curve.jpg "learning curve comparison"
[image19]: ./results/web_images_results.jpg "web image result"
[image20]: ./results/top5softmax.jpg "top5softmax"
[image21]: ./results/cf_matrix.jpg "confusion matrix"
[image22]: ./results/image_for_fmap1.jpg "Orignal image 1 for feature map"
[image23]: ./results/feature_map11.jpg "image 1 conv1 feature map"
[image24]: ./results/image_for_fmap2.jpg "Orignal image 2 for feature map"
[image25]: ./results/feature_map21.jpg "image 2 conv1 feature map"
<!-- ## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.   -->

---
<!-- ### Writeup / README -->

<!-- #### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code. -->
Here is a link to my [project code](https://github.com/kaoozhi/CarND-Traffic-Sign-Classifier-Project/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary of the data set:
<!--
I used the pandas library to calculate summary statistics of the traffic
signs data set: -->

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of images of each class in the training dataset. I noticed that the distribution of classes is quiet unbalanced. Several classes have more than 700 examples (ex. speed limit (50km/h), speed limit (30km/h), yield...) where several have less than 100 examples (ex. Dangerous curve to the left, End of no passing, Speed limit (20km/h)).

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Data preprocessing

As a first step, I decided to convert the images to grayscale to focus on traffic signs' pattern and accelerate the training process.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I standardized the image data by scaling pixel values to have a zero mean and unit variance.

As the training dataset has very unbalanced samples between classes, I decided to augment artificially the image data such that the training dataset gets a balanced class distribution.

![alt text][image15]

To add more data to the the data set, I used random image transformation combined with horizontal/vertical shift, zoom, shear and rotation.

Here is an example of original image samples:

![alt text][image16]

And for augmented image samples:

![alt text][image17]

Finally, some of the images in the training dataset are taken in low contrast conditions which will prevent the network from seeing all pixels information behind the darkness. I then applied an adaptive histogram equalization on images suffering from low contrast. The following an example of the effect of histogram equalization:

![alt text][image3]

#### 2. Model Architecture

I used the LeNet architecture and my final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| Batch Normalization | |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Dropout      	| 0.2				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x64       									|
| RELU					|												|
| Batch Normalization | |
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x64 				|
| Dropout      	| 0.2				|
| Fully connected		| output 120       									|
| RELU					|												|
| Fully connected		| output 84       									|
| RELU					|												|
| Fully connected		| output 43       									|



#### 3. Model Training

To train the model, I used an ADAM optimizer to minimize the cost function represented by the cross entropy within the following hyperparameters setting:
* batch size = 512
* epochs = 20
* learning rate = 0.002

#### 4. Solution Approach

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 98.5%
* test set accuracy of 95.4%

The model selection was done through an iterative approach:
<!-- * What was the first architecture that was tried and why was it chosen? -->
First architecture tried was the original LeNet-5 model from the class, as it is a well built and efficient architecture for image classification.
<!-- * What were some problems with the initial architecture? -->
The first architecture was fast on training but did not returned satisfied accuracy after several hyperparameters tuning iterations. The results showed an overfitting issue, the model had a good bias level while there was a great gap between training loss and validation loss.

As the input from prior layers can vary after weights updates, I first added a batch normalization right after convolution layer to standardize inputs fed to activation unit, the technique helps stabilize and accelerate the training and offering a second benefit on regularization. I added as well the dropout regularization term to help the model generalize better (break potential dependencies between training data and some nodes), but at the meantime I tried not to degrade much the bias level, so I increased the filter depth to consider more features as well.
My hyperparameters tuning mainly focused on the probability of dropout and the depth of convolution filters to manage the overfitting/underfitting trade-off. I then decreased a bit the learning rate but increased the epochs to reach the final model results.

The final model output a low bias level and kept a reasonable loss gap between training and validation.

![alt text][image18]
### Test a Model on New Images

#### 1. Acquiring New Images

Here are ten German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13]

The images are all taken in good light conditions with a normal contrast level. Those images also have a good sharpness. The 5th image with a "Dangerous to turn right" sign slightly rotated. The 7th image "Slippery road" with watermark which may introduce undesired bias for the model to predict correctly.

#### 2. Predictions on these new traffic signs
<!-- At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric). -->

Here are the results of the prediction:

![alt text][image19]

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.4%

#### 3. Softmax probabilities analysis
<!-- Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts) -->

The code for making predictions on my final model is located in the line #208 of the Ipython notebook.

For each of the images, the top five soft max probabilities were:
![alt text][image20]

The model is quiet sure (probability close to 100%) about what it predicts for the images. Except for the "Slippery road" image and "No entry", the model has less certainty but predicts correctly.

Again by visualizing the confusion matrix of test dataset, we can get a more clear idea on how certain the model predicts on each of the classes:

![alt text][image21]

The model sees the "Speed limit (60km/h)" sign as "Speed limit (80km/h)", "Speed limit (100km/h)" sign as "Speed limit (120km/h)" a lot.
The model has trouble as well to predict "Pedestrians","Beware of ice/snow" and "Bumpy road":

|ClassId| SignName | Precision	|Recall
|:---------------------:|:---------------------:|:---------------------:|---------------------:|
|27	|Pedestrians	|0.882353	|0.500000
|30	|Beware of ice/snow	|0.840909	|0.740000
|22	|Bumpy road	|0.959184|	0.783333
|3|	Speed limit (60km/h)	|0.975069	|0.782222
|7	|Speed limit (100km/h)	|0.990000	|0.880000

The model's certainty about "Double curve" sign and "Speed limit (20km/h)" sign is still limited.

|ClassId| SignName | Precision	|Recall
|:---------------------:|:---------------------:|:---------------------:|---------------------:|
|21	|Double curve	|0.566434	|0.900000
|0	|Speed limit (20km/h)	|0.674699	|0.933333

Further work on image preprocessing and model architecture is still needed to improve model's generalization ability about those classes.
### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Visual output of network's feature maps.
The model has a high precision and recall score for the "Speed limit(30km/h)" sign.
Here are feature map visualization of the first convolution layer output of two "Speed limit(30km/h)" samples

Fisrt image:

![alt text][image22]

1st conv layer output:

![alt text][image23]

Second image:

![alt text][image24]

1st conv layer output:

![alt text][image25]

Filter 3,4,5,8,22,26,27,31 mainly focus on the number's shape, Filter 20 reads the outline of the sign, and the red circle is the common feature for the majority of filters
<!-- 2nd conv layer output:

![alt text][image24] -->
