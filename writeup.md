# **Traffic Sign Recognition**

## Writeup
---

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
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./data/web_images/img01.jpg "Traffic Sign 1"
[image5]: ./data/web_images/img02.jpg "Traffic Sign 2"
[image6]: ./data/web_images/img03.jpg "Traffic Sign 3"
[image7]: ./data/web_images/img04.jpg "Traffic Sign 4"
[image8]: ./data/web_images/img05.jpg "Traffic Sign 5"
[image9]: ./data/web_images/img06.jpg "Traffic Sign 6"
[image10]: ./data/web_images/img07.jpg "Traffic Sign 7"
[image11]: ./data/web_images/img08.jpg "Traffic Sign 8"
[image12]: ./data/web_images/img09.jpg "Traffic Sign 9"
[image13]: ./data/web_images/img10.jpg "Traffic Sign 10"
[image14]: ./results/1st_model_lrn_curve.jpg "1st model results"
[image15]: ./results/augmented_data_visulization.jpg "augmented data"
[image16]: ./results/original_data_sample.jpg "original data sample"
[image17]: ./results/augmented_data_sample.jpg "augmented data sample"
[image18]: ./results/lrn_curve_compa.jpg "learning curve comparison"
[image19]: ./results/web_images_results.jpg "web image result"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kaoozhi/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

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

To add more data to the the data set, I used random image transformation combined with horizontal/vertical shift and rotation.

Here is an example of original image samples:

![alt text][image16]

And for augmented image samples:

![alt text][image17]

#### 2. Model Architecture

I used the LeNet architecture and my final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Average pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Dropout      	| 0.2				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x18       									|
| RELU					|												|
| Average pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 				|
| Dropout      	| 0.2				|
| Fully connected		| output 120       									|
| RELU					|												|
| Fully connected		| output 84       									|
| RELU					|												|
| Fully connected		| output 43       									|



#### 3. Model Training

To train the model, I used an ADAM optimizer to minimize the cost function represented by the cross entropy within the following hyperparameters setting:
* batch size = 512
* epochs = 50
* learning rate = 0.003

#### 4. Solution Approach

My final model results were:
* training set accuracy of 0.976
* validation set accuracy of 0.959
* test set accuracy of 0.949

The model selection was done through an iterative approach:
<!-- * What was the first architecture that was tried and why was it chosen? -->
First architecture tried was the original LeNet-5 model from the class, as it is a well built and efficient architecture for image classification.
<!-- * What were some problems with the initial architecture? -->
The first architecture was fast on training but did not returned satisfied accuracy after several hyperparameters tuning iterations:
* training set accuracy of 0.9876
* validation set accuracy of 0.905
* test set accuracy of 0.902

The results showed an overfitting issue, the model had a good bias level while there was a great gap between training loss and validation loss.
![alt text][image14]

I first decided to add dropout regularization term to help the model generalize better (break potential dependencies between training data and some neurons), but at the meantime I tried not to degrade much the low bias level, so I increased the filter depth to consider more features as well. I observed some training images have a very bright background, I then switched to Average pooling layer instead of max pooling, which is more appropriate for the filter to focus on traffic signs' pixels other than bright background's ones.
My hyperparameters tuning mainly focused on the probability of dropout and the depth of convolution filters to manage the overfitting/underfitting trade-off. I then decreased a bit the learning rate but increased the epochs to reach the final model results.

The final model decreased significantly the validation loss (overfitting) and kept the similar bias level.

![alt text][image18]
### Test a Model on New Images

#### 1. Acquiring New Images

Here are ten German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image19]

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the line #549 of the Ipython notebook.

For all of the 10 images, the model is quiet sure for what it predicts (max probability > 0.99). The top five soft max probabilities were:

1st image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
|9.999827e-01 |	Children crossing
|1.717114e-05	| Beware of ice/snow
|1.277206e-07 |	Slippery road
|1.153143e-09	| Dangerous curve to the left
|4.037057e-10	| Right-of-way at the next intersection

2nd image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
| 1.000000e+00 | Right-of-way at the next intersection
|  2.419150e-13   |                  Beware of ice/snow
|  4.862635e-16 |                           Pedestrians
| 3.491030e-18  |                         Double curve
|  1.298773e-18  |                        Slippery road

3rd image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
|  1.000000e+00    |                  Speed limit (70km/h)
|  3.932104e-08    |                  Speed limit (20km/h)
|  2.123625e-11    |                  Speed limit (30km/h)
|  2.411478e-14    |                  Speed limit (60km/h)
| 4.749064e-18 | Vehicles over 3.5 metric tons prohibited

4th image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
|     0.999971    |               Stop
|    0.000019  | Speed limit (70km/h)
|     0.000006   |Speed limit (60km/h)
|      0.000003 |  Speed limit (20km/h)
|    0.000001 | Speed limit (120km/h)

5th image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
| 1.000000e+00     |      Dangerous curve to the right
|  3.515676e-13   |         Dangerous curve to the left
| 4.354769e-15   |                       Slippery road
|  3.838899e-20 |                       General caution
| 2.925018e-20 | Right-of-way at the next intersection

6th image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
|  9.999667e-01 | End of all speed and passing limits
| 3.315090e-05  |                  End of no passing
| 8.533422e-08 |                     General caution
|  4.111942e-08 |                          No passing
|  3.249423e-11  |                              Yield

7th image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
|    0.998706     |     Slippery road
|   0.000928 |    Beware of ice/snow
|    0.000363  |    Bicycles crossing
|   0.000002  |    Children crossing
|    0.000002 | Wild animals crossing

8th image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
|  9.960186e-01                                   | No passing
|  3.981337e-03           |                  End of no passing
| 7.146467e-08      |Vehicles over 3.5 metric tons prohibited
|  3.239753e-09  |No passing for vehicles over 3.5 metric tons
| 1.536162e-09       |    End of all speed and passing limits

9th image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
| 9.999954e-01         |                    No entry
| 4.660556e-06         |           End of no passing
| 9.051950e-13  |End of all speed and passing limits
| 2.802570e-13           |                      Stop
|  1.939802e-14         |        Speed limit (20km/h)

10th image:

| Probability | Prediction	        					|
|:---------------------:|:---------------------------------------------:
| 1.000000e+00      |  Turn left ahead
|  4.151513e-12      |       Ahead only
|  1.393350e-15    |  Bicycles crossing
| 7.461991e-16       |   Priority road
|  7.585137e-17  |Speed limit (120km/h)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
