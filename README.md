README.md
kmeans_openCV branch contains a notebook which pulls images from the ML Project Google Drive folder: Forest fire images.
It iterates through the images, clusters their pixels using the cv2 kmeans function, and shows the image with its pixels confined to those clusters.
The variable K defines how many clusters to use.
In the last cell of the notebook, you can test it out on a single image "example2.jpg" as long as that image is stored in the same place as the code. I ran this in Google Colaboratory (openCV does not work well with Jupyter apparently), so I clicked on the folder icon on the far left and dragged a jpg that I had named "example.jpg" into that sidebar. Then I test out the function at the bottom.
Once that is working, run the rest of the cells from top to bottom, and feel free to experiment with the value K.
