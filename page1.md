Problem Statement: How do we most accurately model and predict the spread of wildfires in a given region using machine learning techniques?

<a href="index.md">Midterm Report</a>

Data Sources: NASA Wildfire Data, UCI Forest Fire Data, Kaggle Forest Fire and Wildfire Data
Data: 
UCI Forest Fire dataset: https://archive.ics.uci.edu/ml/datasets/Forest+Fires

Kaggle datasets: https://www.kaggle.com/elikplim/forest-fires-data-set
		    https://www.kaggle.com/rtatman/188-million-us-wildfires

NASA has data: https://earthdata.nasa.gov/learn/toolkits/wildfires

Introduction: 
There were 50,477 wildfires in 2019 and 58,083 wildfires in 2018, according to the National Interagency Fire Center (NIFC). Around 4.7 million acres were burned in 2019 and  8.8 million acres burned in 2018. Wildfires are of high exigence for the damage they do to wildlife and air quality, and the threat they pose to human lives and nature preservation.

In our project proposal we set out to use NASA data to help in the prediction and analysis of forest fires. For our second touchpoint we will be discussing our unsupervised learning techniques, including how we used interpretations of the expectation–maximization and kmeans algorithms to cluster regions within our chosen satellite images, and how we plan to use the patterns found in the satellite images as a pre-processing step for the supervised portion of our project.

Methods:

Inspired by a reading of a 2019 paper on Segmentation of Fire and Smoke from Infra-Red Videos and a 2013 paper on flame segmentation based on flame pixel identification, we wanted to use clustering to find patterns in images to aid in the detection of forest fires. 

Our input data was a series of labelled images from the NASA MODIS sensor on the Aqua and Terra satellite. The imagery from these satellites has a high resolution, can be downloaded in the form of RBG Jpegs, making them ideal for the kind of image analysis we wanted to perform on them. Their sizes were standardized, and we selected a portion of images from our dataset to test our unsupervised learning algorithms. We tried to test on a range of images that features fires and/or smoke, or were just regular satellite images that did not include any kind of fire-related natural disasters.

We used EM and Kmeans to cluster and manipulate our images, firstly trying to cluster by warm tones featured in fires, but also by the color and directional component of smoke, as smoke has a distinct directional component (like a straight streak) that differentiates it from fog or sand storms. 

Discussion/ Challenges
As we have not yet started the supervised stage of our project, our main challenge has been to get the images clustered in a way that makes classifying them as either photos of forest fires or not the easiest it can be. In addition, trying to process and find smoke in images proved to be somewhat difficult and error prone. Another somewhat less pressing issue is the difference between forest fires and other natural phenomenon such as volcanic eruptions, or intentional burning practices.

As you can see, here are some of the examples of clustering algorithms we have run on our NASA satellite images.

We hope that the visual differences and contrast between elements that could reasonably be fire or smoke and other elements in the images will help us to classify the images with a boolean output in the supervised learning portion of our project, which we are hoping to do using a neural net.






