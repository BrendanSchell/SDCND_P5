
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image6]: ./output_images/windowed_image.png
[image5]: ./output_images/heat_map_image.png
[image4]: ./output_images/final_image.png
[video1]: ./project_video_out.mp4



## Histogram of Oriented Gradients (HOG)

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I began by reading in all of the vehicle and non-vehicle images from the dataset folders. This step can be found in the second cell of the Jupyter notebook.
Images of the two classes can be seen below.

![alt text][image1]

I then tested various color spaces and the use of different 'skimage.hog()' parameters for 'orientations', 'pixels_per_cell', and 'cells_per_block'.
Random images of each of the two classes were selected with their HOG features plotted in order to visualize the data.

All channels of the 'YUV' space were used and HOG parameters with orientations=6, pixels_per_cell=(8, 8) and cells_per_block=(2, 2) implemented.
Here is an example of images plotted for both cars and non-cars using the selected parameters. The extraction of hog features is performed by the 'get_hog_features()' function
in cell 3 of the Jupyter notebook.

![alt text][image2]

### 2. Explain how you settled on your final choice of HOG parameters.

Parameters were first selected in order to produce an image pipeline that resulted in reasonable results. Originally, parameters of 9, (8,8), and (2,2) were
used for the orientations, pixels_per_cell, and cells_per_block respectively. In order to reduce the runtime of the classifier, the orientations were reduced from 9 to 6 since it did not result in a large drop in classifier accuracy. The YUV colour space was used with all channels since it yielded a very good validation accuracy
with this dataset (> 98%). The space and colour features were also included for use in the classifier since they did not take significantly longer to execute
and resulted in slightly better validation accuracy. The parameters were mostly trained through intuition and a trial and error approach.

### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For this project, a linear SVM was trained since it was easy to implement using SKLearn, fast to train and predict, and resulted in a very good validation accuracy for this dataset. Only the included
dataset with no data augmentation was used since it was sufficient to achieve a good quality result.
The classifier was trained using the 'train_classifier()' function in cell 4 of the Jupyter notebook. This function begins by extracting the features from both the car and not-car datasets using the
'extract_features()' function. 'extract_features()' converts the image to the YUV colour space, and extracts the spatial features, colour features (using a histogram of colours), and finally the HOG features
using the parameters mentioned previously. The features were appended together and normalized using SKLearn's StandardScaler function to ensure that all of the features were on the same scale. The features and 
labels were split into train and test sets using a 80/20 train/test split. SKLearn's linear SVM classifier was then used with no hyperparameter tuning to fit the model with a 0.9859 validation accuracy.

## Sliding Window Search

### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A sliding window search was performed using the 'slide_windows()' function in cell 3 of the Jupyter notebook and implemented in cell 6 of the 'image_pipeline()' function. 
The implementation uses two window sizes of (128,128) and (96,96). The larger window size is used for the lower portion of the image while the smaller window size is used to 
detect vehicles that are a further distance from the camera. This is done since the vehicles appear to be smaller at further distances in the image. The image search space
was narrowed in order to reduce computation time and eliminate false positives (or true positives if in the opposite lane) from appearing. The x range was narrowed
to search for any windows that appear more than 700 pixels from the left of the image while the y range was narrowed to search anywhere more than 400 pixels from the top of the image.
Overlap values of 0.9 and 0.2 were used for the x and y ranges respectively. It was found that vehicles tended to vary significantly in the horizontal direction, but not very much vertically 
so these values were used to account for this. An example can be seen below of the windows that were searched (note that only 1 / 4 of the windows are displayed to appear cleaner).

![alt text][image3]

### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In order to optimize the performance of the classifier, I tested various colour spaces along with the inclusion / exclusion of different types of features.
I did not use any hyperparameter tuning or data augmentation since I found them to be unecessary to achieve good results for the task.
Through trial and error, I achieved great performance using spatially binned colour, histograms of colour, and HOG features with all colour channels in the YUV colour space.
In order to optimize the prediction time of the classifier, I did reduce the number of orientations to use for the HOG features.

Some example images of the final pipeline can be seen below.

![alt text][image4]
---

## Video Implementation

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here is a [link to my video result](./project_video_out.mp4)


### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Positive detections were first found using the 'search_windows()' function in cell 3 of the Jupyter notebook. These positive detections with overlaps were then passed to the 'add_heat()' function which
added a unit of heat to the pixels of each window appearing in the positive vehicle detections. Pixels with overlapping windows had more 'heat' and so were more certain to be valid predictions. The heat map
was thresholded to only keep pixels with heat values >= 2 since this resulted in both good retention of true positives while eliminating most false positives.
'scipy.ndimage.measurements.label()' was then used to determine clusters in the heat map and labels were assigned to these clusters corresponding to detected vehicles. Finally, bounding boxes were
placed over the clusters using the 'draw_labeled_bboxes()' function in cell 3 of the Jupyter notebook.


### Here is a frame with its corresponding heat map:

![alt text][image5]

### Here the resulting bounding boxes are drawn onto the frame:
![alt text][image6]



---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took for this project was to start off with the helper functions provided by Udacity through the course material and to use these in a Jupyter notebook to analyze and 
visualize the dataset. I modified these functions and added new functionality in order to produce a rough start-to-end functioning pipeline for the project.
I then modularized individual functions and created helper functions to assist in testing individual parts and the overall start-to-end result of the pipeline.
Afterwards, I tested various parameters and features to use for the classifier in order to achieve high validation accuracy with few false positives. I then used a trial and error approach
to determine parameters for the sliding windows and heat map thresholding that achieved fewer false positives while retaining true positive predictions. Once the image pipeline was functional,
I tested on video and repeated testing until good performance was achieved on the project video. In order to smooth the performance of the detections and reduce time to produce the video,
new detections are only added every five frames if previous detections are available.

Though the pipeline is functional for the project video, there are some areas where it fails. First of all, though there are very few false positives and false negatives, there are some minor occurences
of these in the project video. As well, the detection bounding boxes still appear erratic, even with detections only being done every five frames. Only the right portion of the image is being searched
in this implementation so any video or images taken with cars directly in front or to the left of the camera would not function correctly. The video pipeline is also prone to errors due to changes of
elevation in the video. Finally, the pipeline itself does take time to execute and is not fast enough to work in a real-time environment.

There are some areas of improvement that could be used to make the project more robust. First of all, smoothing using the mean of bounding boxes might be used in order to smooth the appearance of detections.
More testing could be performed on various other colour spaces, channels, feature inclusion / exclusion, data augmentation, Principal Component Analysis, or otherwise in order to improve the classification and reduce the
time to produce predictions. A different classifier such as Ensemble methods, Decision Trees, or Deep Learning could also be used and might be possible improvements over a Linear SVM. Other computer vision techniques
such as perspective transforms might also prove useful in improving detections.

