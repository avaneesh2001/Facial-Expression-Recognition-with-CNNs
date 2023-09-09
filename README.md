# Facial Expression Recognition with CNNs
Facial Expression Recognition (FER) is the process of detecting and classifying a person's emotional state based on facial expressions using image processing techniques. FER has a wide range of applications, including mental health diagnosis, marketing research, and human-computer interaction.

## Dataset

For this project, I utilized the 'Face expression recognition' dataset from Kaggle. This dataset contains grayscale images of size 48x48, representing seven different emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise.

- Training Set: 28,821 images
- Testing Set: 7,066 images

The dataset is organized into seven subfolders, one for each emotion category. This structure makes it convenient to use TensorFlow's 'flow from directory' method for data loading.

## Model Architecture

### CNN Architecture Selection

My goal in selecting the model architecture was to achieve the best validation and test accuracy without overfitting the training dataset. To achieve this, I conducted experiments by training numerous models, varying the following parameters:

1. Number of Convolution Stages.
2. Number of Kernels for each Conv2D layer.
3. Number of Fully Connected/Dense layers.
4. Number of Neurons for each Fully Connected/Dense layer.

#### Choosing the Number of Convolution Stages

In this experiment, I kept the number of Fully Connected layers fixed at 2, with each layer having 128 neurons. I trained the model four times, each time with a different number of Convolution stages. Each Conv2D layer had 64 kernels in all three iterations.

Based on the results, the model with 4 convolution stages achieved the highest validation accuracy without overfitting.

#### Choosing the Number of Fully Connected/Dense Layers

For this experiment, I kept the number of Convolution stages at 4, with the first two Conv2D layers having 64 kernels and the last two having 128 kernels. I trained the model four times, each time with a different number of Fully Connected layers. Each Fully Connected layer had 256 neurons in all four iterations.

Based on these results, the model with 2 Fully Connected layers achieved the highest validation accuracy without overfitting.

#### Choosing the Number of Kernels for Conv2D Layers and Neurons for Fully Connected Layers

I conducted experiments by varying the number of kernels in Conv2D layers and the number of neurons in Fully Connected layers. The loss, accuracy, val loss, and val accuracy of each model were used to plot the results.

From the comparisons, Model 5 offered the best validation accuracy without excessive overfitting, making it the preferred model architecture for my Facial Expression Recognition task.

### Decision to Use a CNN Model

Initially, I experimented with an Inception CNN architecture to potentially improve model accuracy. However, I encountered overfitting issues despite implementing techniques such as data augmentation, normalization, early stopping, and dropout.

Given the persistent overfitting problem, I made the decision to switch to a conventional CNN model architecture. CNNs have proven effective in various computer vision tasks, including Facial Expression Recognition.

### Dataset Consideration

In addition to experimenting with different model architectures, I also explored the use of the 'FER2013' (Facial Expression Recognition 2013) dataset available on Kaggle. It contains images along with categories describing the emotion of the person in it. The dataset contains 48×48 pixel grayscale images with 7 different emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories. The images for 'test' data were pulled from this dataset.


**Note:** If you prefer not to run the project locally, you can also view a read-only copy on Kaggle by [clicking here](https://www.kaggle.com/ravaneesh/cv-project-1). This allows you to explore the project without the need for local setup.


