# Projekt-Seminar Deep Learing (Prof. Dr. Jörn Hees) #
## Group: Leon Schirra, Alexander Kolb, Dominik Zoric, Henrik Theisen ##
## Topic: Finger Counter ##

Our group focused on AI-based hand gesture recognition, employing four distinct approaches, all rooted in the same preparation process.
As part of the preparation, each member captured 100 images for various hand poses, including "0,1,2,3,4,5,Spock,Other." You count along the hand starting with the thumb and ending with the little finger. These images were then mixed and divided into three sets: 80% for training, 10% for testing, and 10% for validation. For each of these three sets, a Mediapipe hand recognition AI was applied, generating CSV files. Each CSV file contains the image names and 21*3 landmark coordinates (21 landmarks with x, y, and z coordinates each). The images and the CSVs can be found in our Dataset Repository: https://github.com/henthe/DatasetForFingerCounter.git
Based on this shared preprocessing, we divided our team's focus into the following topics:
Leon used the 3D coordinates with Tensorflow.
Alexander went with ScikitLearn.
Henrik used LightGBM. Dominik used Mediapipe.

As a second project, Dominik went with a different preprocessing algorithm, he used two-dimensional normalized coordinates and trained his model with the Tensorflow Framework.

The Mediapipe code to generate the landmarks is based on this Repository https://github.com/nicknochnack/MediaPipeHandPose

# Requirements
- Python 3.10.x
- All other required packages can be installed via the Jupyter Notebooks
- Further requirements for 2D_approach in 2D_approach/2DapproachDocu.md


# 2D approach with Tensorflow
To run the "2D_approach" a detailed instruction can be found in ["2D_approach/2approachDocu.md"](https://github.com/henthe/fingerCounter/blob/main/2D_approach/2DapproachDocu.md). The folder contains a jupyter notebook file, 
where you can find the background of the models building, training and Hyperparameter-tuning and as well as an exported TensorFlow model(model.keras), 
which can be tested in real time, capturing your webcam. In this approach, Dominik created his "own" preprocessing algorithm, inspired by an
already existing GitHub repository(detailed information in 2approachDocu.md). He generated two-dimensional landmarks from the pictures, which
the team split beforehand uniformly in train, test and validation.
The model is trained to classify hand gestures into our 8 predefined classes, and the jupyter notebook provides functionality 
for one-time model building, hyperparameter tuning, model evaluation, and visualization of misclassified samples.

### Sources
- [original respository , used as tutorial](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
- [Youtube Tutorial, original repository explained](https://www.youtube.com/watch?v=a99p_fAr6e4&t=157s&ab_channel=IvanGoncharov)
- [GitHub: Google Research - Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Keras Documentation](https://keras.io/keras_tuner/)

# Mediapipe
The "mediapipe" folder contains a Google colab notebook file, 
where you can find the background of the models building, training and an exported model. Keep in mind for the use of the model you need the pictures located somewhere,
where they can be referenced. For further information of the specific folder structure visit: https://developers.google.com/mediapipe/solutions/customization/gesture_recognizer.
After setting up your environment and the pictures, you can execute the cells one by one.
The model is trained to classify hand gestures into our 8 predefined classes, and the jupyter notebook provides functionality 
for one time model building and model evaluation.

### Sources
- [Mediapipe](https://mediapipe.dev/)
- [Mediapipe: Customize Handgestures](https://developers.google.com/mediapipe/solutions/customization/gesture_recognizer)

# LightGBM
LightGBM is a tree based Neural Network Model. This notebook is based on the following notebook:
https://www.kaggle.com/code/bhavinmoriya/lightgbm-classifier-in-python

## Guide
1. Pull the DataSet Repo somewhere
2. Put path to repo in path variable
3. Run everything but the last two Cells

4. For single picture tryout run second to last cell (line 10 image_path has to be the path to the image you want to try)

5. For live demo run last cell

### Sources

- [Kaggle: Tutorial](https://www.kaggle.com/code/prashant111/lightgbm-classifier-in-python)
- [Flaml: Documentation](https://microsoft.github.io/FLAML/docs/getting-started)
- [LightGBM: Documentation](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html)

# Scikit-Learn
1. Pull the DataSet Repo somewhere
2. Put path to repo in path variable
3. Run everything

4. If you only want to execute certain classifiers, only execute the corresponding cells. To use Pipeline then change the parameters of the classifiers to be used.
   
### Sources
- [Scikit-Learn](https://scikit-learn.org/)
- [Kaggle: k-Nearest Neighbors](https://www.kaggle.com/code/amolbhivarkar/knn-for-classification-using-scikit-learn)
- [GitHub: Pipeline & Other Classifier](https://github.com/RDFLib/graph-pattern-learner/blob/master/fusion/trained.py#L194)
- [GitHub: param-grids](https://github.com/RDFLib/graph-pattern-learner/blob/master/fusion/trained.py#L474)

# TensorFlow
The Folder "TensorFlow" contains a jupyter notebook file and a pre-trained TensorFlow model (.keras).
The Python script implements a deep learning model for hand gesture recognition based on landmark data extracted from images. The model is trained to classify hand gestures into predefined classes, and the script provides functionality for hyperparameter tuning, model evaluation, and visualization of misclassified samples.

### Usage

1. Clone the dataset repository if you want to work with our data.
2. Clone this repository and create a new virtual environment to run the Jupyter Notebook.
3. In the first cell of the Jupyter Notebook, adjust the path variables.
4. Run the cells sequentially (most cells require the execution of the previous cell to run).

### Sources

- [YouTube: TensorFlow 2.0 Complete Course - Python Neural Networks for Beginners Tutorial](https://www.youtube.com/watch?v=WVOMGekzbWE&t=2378s)
- [YouTube: Hand Gesture Recognition using TensorFlow | Hand Pose Estimation | MediaPipe](https://www.youtube.com/watch?v=_c_x8A3mNDk&t=7s)
- [GitHub: Google Research - Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Keras Documentation](https://keras.io/keras_tuner/)
  
# License

TODO 
