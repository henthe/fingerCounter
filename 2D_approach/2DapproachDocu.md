# hand-gesture-recognition-using-mediapipe
Estimate hand pose using MediaPipe (Python version).<br> This is a sample 
program that recognizes hand signs and finger gestures with a simple MLP using the detected key points.
<br> ❗ _️**This is English Translated version of the [original repo](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe). All Content is translated to english along with comments and notebooks**_ ❗
<br> 
!
This repository contains the following contents.
* Sample program, by Alex, Leon, Henrik and Dominik
* Hand sign recognition model(TFLite)
* Finger gesture recognition model(TFLite)
* Learning data for hand sign recognition and notebook for learning
* Learning data for finger gesture recognition and notebook for learning

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

# Demo
Here's how to run the demo using your webcam.
```bash
python app.py
```

The following options can be specified when running the demo.
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Height at the time of camera capture (Default：540)
* --use_static_image_mode<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.7)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.7)

# Directory
<pre>
│  app.py #improvements by @Dominik Zoric
   generate_landmarks.py #Author @Dominik Zoric, file is needed to preprocess Images
│  keypoint_classification_EN.ipynb #used for training the model, improvements by @Dominik Zoric
│  point_history_classification.ipynb #useless for usecase "deep learning"
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  │  keypointsTest.csv #Used for testing the model
│  │  │  keypointsTrain.csv #Used for training the model
│  │  │  keypointsVal.csv #Used to optimize Paramtertuning
│  │  └─ keypoint_classifier_label.csv #all data combined not in use for the actual model
│  │          
│  └─point_history_classifier #not used for "deep learning" course
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>

### keypoint_classification_EN.ipynb
This is a model training script for hand sign recognition.


### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypointsTrain.csv)
* Testing data(keypointsTest.csv)
* Validation data(keypointsVal.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)



### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training
Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

### Hand sign recognition training
#### 1.Learning data collection
Use the generate_landmarks.py file to preprocess your pictures. You need to change the path of your pictures manually, 
since we divided our dataset uniformly for each classifier, to achieve comparibilty between each model. 
Optionally the author of the original repo provided a script:
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>


#### 2.Model training
Open "[keypoint_classification_EN.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute the cells for your use case
If you want to run it without Parametertuning, run all cells under the headline "
Old Model building and training without Hyperparameter-tuning". For Hyperparameter-tuning and better accuracy results, run the cells after the Headline 
"Optimized Model building and Paramtertuning". .<br>
Since in our project we have eight classed"NUM_CLASSES = 8" <br><br>

#### X.Model structure


# Reference
* [MediaPipe](https://mediapipe.dev/)

# Original Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)

# Translation and other improvements
Nikita Kiselov(https://github.com/kinivi)

# Further Improvements in context for the "deep Learning" course
Dominik Zoric(https://github.com/domzoric)
 
# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
