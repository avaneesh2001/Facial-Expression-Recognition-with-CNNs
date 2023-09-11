# Facial Expression Recognition with CNNs
Facial Expression Recognition (FER) is the process of detecting and classifying a person's emotional state based on facial expressions using image processing techniques. FER has a wide range of applications, including mental health diagnosis, marketing research, and human-computer interaction.

## Dataset

For this project, I utilized the 'Face expression recognition' dataset from Kaggle. This dataset contains grayscale images of size 48x48, representing seven different emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise.

- Training Set: 28,821 images
- Testing Set: 7,066 images

The dataset is organized into seven subfolders, one for each emotion category. This structure makes it convenient to use TensorFlow's 'flow from directory' method for data loading.

<img src = "https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/face%20emotion%20recognition.jpg" width = 750>

## Model Architecture

### CNN Architecture Selection

My goal in selecting the model architecture was to achieve the best validation and test accuracy without overfitting the training dataset. To achieve this, I conducted experiments by training numerous models, varying the following parameters:

1. Number of Convolution Stages.
2. Number of Kernels for each Conv2D layer.
3. Number of Fully Connected/Dense layers.
4. Number of Neurons for each Fully Connected/Dense layer.

#### Choosing the Number of Convolution Stages

In this experiment, I kept the number of Fully Connected layers fixed at 2, with each layer having 128 neurons. I trained the model four times, each time with a different number of Convolution stages. Each Conv2D layer had 64 kernels in all three iterations.
```
1. Result for the model with 2 Convolution stages:
loss: 1.0406 - accuracy: 0.6082 - val loss: 1.1011 - val accuracy: 0.5917 - test loss: 0.9450 - test accuracy:
0.6591
2. Result for the model with 3 Convolution stages:
loss: 0.9832 - accuracy: 0.6314 - val loss: 1.0303 - val accuracy: 0.6162 - test loss: 0.8786 - test accuracy:
0.6748
3. Result for the model with 4 Convolution stages:
loss: 1.0132 - accuracy: 0.6142 - val loss: 1.0071 - val accuracy: 0.6281 - test loss: 0.9124 - test accuracy:
0.6548
4. Result for the model with 5 Convolution stages:
loss: 1.0619 - accuracy: 0.5969 - val loss: 1.0216 - val accuracy: 0.6199 - test loss: 0.9433 - test accuracy:
0.6429

```

Based on the results, the model with 4 convolution stages achieved the highest validation accuracy without overfitting.

#### Choosing the Number of Fully Connected/Dense Layers

For this experiment, I kept the number of Convolution stages at 4, with the first two Conv2D layers having 64 kernels and the last two having 128 kernels. I trained the model four times, each time with a different number of Fully Connected layers. Each Fully Connected layer had 256 neurons in all four iterations.
```
1. Result for the model with 1 Fully Connected layers:
loss: 0.8841 - accuracy: 0.6712 - val loss: 0.9904 - val accuracy: 0.6386 - test loss: 0.8029 - test accuracy:
0.7086
2. Result for the model with 2 Fully Connected layers:
loss: 0.8847 - accuracy: 0.6669 - val loss: 0.9805 - val accuracy: 0.6477 - test loss: 0.7874 - test accuracy:
6
0.7137
3. Result for the model with 3 Fully Connected layers:
loss: 1.0163 - accuracy: 0.6139 - val loss: 1.0363 - val accuracy: 0.6115 - test loss: 0.9139 - test accuracy:
0.6623
4. Result for the model with 4 Fully Connected layers:
loss: 1.0252 - accuracy: 0.6142 - val loss: 1.0471 - val accuracy: 0.6095 - test loss: 0.9322 - test accuracy:
0.6523
```

Based on these results, the model with 2 Fully Connected layers achieved the highest validation accuracy without overfitting.

#### Choosing the Number of Kernels for Conv2D Layers and Neurons for Fully Connected Layers

I conducted experiments by varying the number of kernels in Conv2D layers and the number of neurons in Fully Connected layers. The loss, accuracy, val loss, and val accuracy of each model were used to plot the results.

![Comparision](https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/table.png)

From the comparisons, Model 5 offered the best validation accuracy without excessive overfitting, making it the preferred model architecture for my Facial Expression Recognition task.
### Block Diagram
<img  src="https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/1.png" width = 850>
*Block Diagram*

### Decision to Use a CNN Model

Initially, I experimented with an Inception CNN architecture to potentially improve model accuracy. However, I encountered overfitting issues despite implementing techniques such as data augmentation, normalization, early stopping, and dropout.
![Inception Results](https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/table%20inception.png)
*Results*
<img src="https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/2.png" width = 850>
*Block Diagram*

Given the persistent overfitting problem, I made the decision to switch to a conventional CNN model architecture. CNNs have proven effective in various computer vision tasks, including Facial Expression Recognition.

### Dataset Consideration

In addition to experimenting with different model architectures, I also explored the use of the 'FER2013' (Facial Expression Recognition 2013) dataset available on Kaggle. It contains images along with categories describing the emotion of the person in it. The dataset contains 48×48 pixel grayscale images with 7 different emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories. The images for 'test' data were pulled from this dataset.
![Dataset samples](https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/fer%20samples.jpeg)


## Performance Metrices
### Model Evaluation with Test Data
```
1795/1795 [==============================] - 13s 7ms/step - loss: 0.7016 - accuracy: 0.7529
```
### Plots
![Accuracy and Loss:Inception Trained on FER dataset](https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/__results___3_0.png)
*Accuracy and Loss:Inception Trained on FER dataset*
![Accuracy and Loss:Inception Trained on FER dataset with augmentation](https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/__results___3_1.png)
*Accuracy and Loss:Inception Trained on FER dataset with augmentation*
![https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/__results___3_1.png](https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/__results___3_2.png)
*Accuracy and Loss:Inception Trained on Facial Emotion Recognition dataset with augmentation*
![ Accuracy and Loss:CNN Trained on FER dataset](https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/__results___3_3.png)
*Accuracy and Loss:CNN Trained on FER dataset*
![https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/__results___3_3.png](https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/__results___3_4.png)
*Accuracy and Loss:CNN Trained on FER dataset with augmentation*

### Confusion Metrix
<img src="https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/confusion%20matrix.png" width = 700)
*Confusion Matrix*

### Model Outputs


<img src="https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/output%201.png" width="450" />
<img src = "https://github.com/avaneesh2001/Facial-Expression-Recognition-with-CNNs/blob/main/Images/output.png" width = "450">


**Note:** If you prefer not to run the project locally, you can also view a read-only copy on Kaggle by [clicking here](https://www.kaggle.com/ravaneesh/cv-project-1). This allows you to explore the project without the need for local setup.


