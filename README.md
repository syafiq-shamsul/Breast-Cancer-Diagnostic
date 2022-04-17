# Breast-Cancer-Diagnostic

## 1. Summary
THe project is to perform diagnostic for breast cancer dataset using feedfoward neural network. The dataset used for the model is the Winsconsin Breast Cancer Dataset. The model created is to predict whether the cancer is benign or malignant

## 2. IDE and Framework
IDE - Spyder
Frameworks - TensorFlow Keras, Pandas, Numpy, Matplotlib, & Scikit-learn

## 3. Methodology

## Data Pipeline
The unrelated columns of features from the data set will be removed initially. The label will be encoded to binary format in terms of 0 and 1. The data will be split with the ratio of 80:20 for features will be rescaled.

## Model Pipeline
The model is trained using a simple feedfoward neural network. Dropout and regularization is applied to prevent the model to overfit. The model structure is visualized as in figure below
![mod![Accuracy](https://user-images.githubusercontent.com/100821053/163700664-254397c8-7d85-4834-bd57-99bdb9f1ae00.png)
el structure](https://user-images.githubusercontent.com/100821053/163700636-90300242-3b50-49ad-92af-6242867be2ce.png)

The model is trained with batch size of 16 and epochs of 40 resulting both training and validation accuracy of 98%. The figure below visualize the loss and accuracy during the training process
![Loss](https://user-images.githubusercontent.com/100821053/163700663-d570e14c-0da4-4307-9e1f-33e00a716efb.png)
![Accuracy](https://user-images.githubusercontent.com/100821053/163700780-3418e775-bd23-49a5-83fe-aa34a1d622d7.png)

## 4. Results

The model is tested using the test data and resulting the accuracy as shown in figure below:
![test results](https://user-images.githubusercontent.com/100821053/163700719-c0e3dfcf-2498-40d6-a332-ba08f05e7113.png)

The figure shows the predictions result for all the test result

![Prediction result](https://user-images.githubusercontent.com/100821053/163700721-ac7f226b-f597-4618-a60e-d966162b0ca4.png)




