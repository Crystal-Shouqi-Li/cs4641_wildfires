Problem Statement: How do we most accurately model and predict the spread of wildfires in a given region using machine learning techniques?

<a href="index.md">Midterm Report</a>

**Data Sources: NASA Wildfire Data, UCI Forest Fire Data, Kaggle Forest Fire and Wildfire Data**

**Introduction:**
There were 50,477 wildfires in 2019 and 58,083 wildfires in 2018, according to the National Interagency Fire Center (NIFC). Around 4.7 million acres were burned in 2019 and  8.8 million acres burned in 2018. Wildfires are of high exigence for the damage they do to wildlife and air quality, and the threat they pose to human lives and nature preservation.

In our project proposal we set out to use NASA data to help in the prediction and analysis of forest fires. For our second touchpoint we will be discussing our unsupervised learning techniques. We implemented two algorithms(expectation–maximization and kmeans) and applied them to the images in order to preprocess and detect the features. Along with clustering them into two groups in order to try and see their accuracy in seperating out images with and without fire. We plan to use the algorithms and results found in the satellite images as a pre-processing step for the supervised portion of our project.

**Methods:**

Inspired by a reading of a 2019 paper on Segmentation of Fire and Smoke from Infra-Red Videos and a 2013 paper on flame segmentation based on flame pixel identification, we wanted to use clustering to find patterns in images to aid in the detection of forest fires. 

**DATA:**

Our input data was a series of labelled images from the NASA MODIS sensor on the Aqua and Terra satellite. We chose these images from satellites since they have a high resolution and can be downloaded in the form of RBG Jpegs. This made them ideal for the kind of image analysis we wanted to perform on them. The images ranged from featuring fires and/or smoke to regular satellite images that did not include any kind of fire-related natural disasters. 
![Image](forestFireDataSet.png)
![Image](notForesetFire.png)

**PROCESS:**
In order to be able to feed the images through the algorithms, we needed to reduce the images into numbers.To start we standardized the images so that they were all the same size and then selected a portion of images from our dataset to run our unsupervised learning algorithms. We didn't use all of the data set in order to save computation power.

We then used EM and Kmeans to segment the colors in our images, firstly trying to cluster by warm tones featured in fires, but also by the color and directional component of smoke, as smoke has a distinct directional component (like a straight streak) that differentiates it from fog or sand storms. 

**ORIGINAL:(manitoba)**
![Image](origManitoba.png) 
**AFTER: (applied EM)** ![Image](manitobaEM.png)

**After Applying two different techniques of Kmeans**

![Image](kMeans2.png)
![Image](kMeans1.png)

In our first attempt to apply Kmeans to our images we used the pixel colors as our features. As a result, when we clustered the images into two clusters (to simulate the binary fire vs not fire classifcation) we got two unequal clusters that were worse than just randomly assigning half the data into one cluster and the other half into another. 

![Image](badCluster0.png)
![Image](badCluster1.png)


**Discussion/ Challenges**
As we have not yet started the supervised stage of our project, our main challenge has been to get the images clustered in a way that makes classifying them as either photos of forest fires or not the easiest it can be. In addition, trying to process and find smoke in images proved to be somewhat difficult and error prone. Another somewhat less pressing issue is the difference between forest fires and other natural phenomenon such as volcanic eruptions, or intentional burning practices.

As you can see, here are some of the examples of clustering algorithms we have run on our NASA satellite images.

We hope that the visual differences and contrast between elements that could reasonably be fire or smoke and other elements in the images will help us to classify the images with a boolean output in the supervised learning portion of our project, which we are hoping to do using a neural net.






