# DS-577-Assignment-2

## Q1. Linear Regression

In this project, you will analyze the privacy leakages caused by Google data collection program. Google Maps collects a significant amount of noisy location data from mobile phones. This data could be used to estimate the future locations of a user. In this project, you are provided with a dataset of a WPI employee collected by Google over a period of 5 weeks.

You are required to use a regression method to estimate the location of the user. You should decide which regression method is most suitable for the given data. Please keep in mind that your data is noisy. Therefore, you should use a regression method to estimate the result and then use rounding to find the most suitable label.

You can train your regression method using the first 4 weeks data. The aim is to estimate the label with a high accuracy for each timestamp in the next weeks. To validate your model, you can use week5.mat data to find the correct label and calculate the estimation accuracy.

Your grade will be determined by testing week6.mat data in your model. Your code should find the correct label and it should give a confidence accuracy. The average accuracy of week6.mat will decide your project grade. Please write your code in Matlab or Python. You are also allowed to use Matlab or Python tools. The code should be commented out and submitted on Canvas since your code will be separately run in these platforms to test week6.mat. The week6.mat data will be available to you after the submission deadline to have an idea on how well your method is working.

The data can be accessed from the canvas page.

## Q2. K-means Clustering

Apply K-means clustering to the same data set and cluster the given locations. You only need longitude and latitude columns (second and third columns). You should apply elbow method clustering to decide optimal K. Please include the plot too. Also note that, K-means clustering is an unsupervised learning. So that you can use 5 weeks data to train your model.
