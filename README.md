Image Classifier using PyTorch
This repository contains a simple image classifier using PyTorch. The classifier is trained on the MNIST dataset and can be used to classify handwritten digits.

Getting Started
Prerequisites
Python 3.x
PyTorch
PIL (Python Imaging Library)
MNIST dataset (downloaded automatically when running the script)
Running the Script
Clone this repository and navigate to the directory.
Run the script using python script.py (assuming the script is named script.py).
The script will download the MNIST dataset, train the classifier, and save the model state to a file named model_state.pt.
Once the training is complete, you can use the classifier to classify images by loading the saved model state and passing an image tensor to the clf object.
Usage
Training the Classifier
The script trains the classifier on the MNIST dataset for 10 epochs with a batch size of 32. You can adjust these hyperparameters by modifying the script.

Classifying Images
To classify an image, load the saved model state using clf.load_state_dict(load(f)) and pass an image tensor to the clf object. The output will be the predicted class label.

For example:
...
img = Image.open('img_1.jpg')
img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
print(torch.argmax(clf(img_tensor)))
...
This will classify the image img_1.jpg and print the predicted class label.

Model Architecture
The classifier uses a convolutional neural network (CNN) with the following architecture:

Conv2d(1, 32, (3,3)) -> ReLU
Conv2d(32, 64, (3,3)) -> ReLU
Conv2d(64, 64, (3,3)) -> ReLU
Flatten
Linear(642222, 10)
The model is defined in the ImageClassifier class.
