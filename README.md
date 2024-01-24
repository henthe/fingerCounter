# fingerCounter
FingerCounting University Deep Learning Project

DataSet Repo: https://github.com/henthe/DatasetForFingerCounter.git

# 2D approach with Tensorflow
To run the "2D_approach" a detailed instruction can be found in "2D_approach/2approachDocu.md". The folder contains a jupyter notebook file, 
where you can find the background of the models building, training and Hyperparameter-tuning and an exported TensorFlow model(model.keras), 
which can be tested in real time, capturing your webcam. In this approach I created my "own" preprocessing algorithm, inspired by an
already existing GitHub repository(detailed information in 2approachDocu.md). I generated two-dimensional landmarks from the pictures, which
we split beforehand uniformly in train, test and validation.
The model is trained to classify hand gestures into our 8 predefined classes, and the jupyter notebook provides functionality 
for one time model building, hyperparameter tuning, model evaluation, and visualization of misclassified samples.

# LightGBM
1. Pull the DataSet Repo somewhere
2. Put path to repo in path variable
3. Run everything but the last two Cells

4. For single picture tryout run second to last cell (line 10 image_path has to be the path to the image you want to try)

5. For live demo run last cell



# TensorFlow

The Folder "TensorFlow" contains a jupyter notebook file and a pre-trained TensorFlow model (.kreas).
The Python script implements a deep learning model for hand gesture recognition based on landmark data extracted from images. The model is trained to classify hand gestures into predefined classes, and the script provides functionality for hyperparameter tuning, model evaluation, and visualization of misclassified samples.


## Requirements

- Python 3.9 - 3.11
- All other required packages can be installed via the Jupyter Notebook


## Usage

1. Clone the dataset repository if you want to work with our data.
2. Clone this repository and create a new virtual environment to run the Jupyter Notebook.
3. In the first cell of the Jupyter Notebook, adjust the path variables.
4. Run the cells sequentially (most cells require the execution of the previous cell to run).

## Sources

- [YouTube: TensorFlow 2.0 Complete Course - Python Neural Networks for Beginners Tutorial](https://www.youtube.com/watch?v=WVOMGekzbWE&t=2378s)
- [YouTube: Hand Gesture Recognition using TensorFlow | Hand Pose Estimation | MediaPipe](https://www.youtube.com/watch?v=_c_x8A3mNDk&t=7s)
- [GitHub: Google Research - Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Keras Documentation](https://keras.io/keras_tuner/)

# License

TODO 