<h1>Image Classifier using PyTorch</h1>

<p>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://www.python.org/downloads/">
    <img alt="Python: 3.x" src="https://img.shields.io/badge/Python-3.x-blue.svg">
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch: 1.x" src="https://img.shields.io/badge/PyTorch-1.x-orange.svg">
  </a>
</p>

<p>A simple image classifier using PyTorch, trained on the MNIST dataset.</p>
<img src = "https://github.com/Sadhurahavan5555/MNIST_Classifier/blob/master/screenshot/MNIST.png">

<h2>Table of Contents</h2>

<ul>
  <li><a href="#getting-started">Getting Started</a></li>
  <li><a href="#prerequisites">Prerequisites</a></li>
  <li><a href="#running-the-script">Running the Script</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#model-architecture">Model Architecture</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#contributing">Contributing</a></li>
</ul>

<h2 id="getting-started">Getting Started</h2>

<h3 id="prerequisites">Prerequisites</h3>

<ul>
  <li>Python 3.x</li>
  <li>PyTorch</li>
  <li>PIL (Python Imaging Library)</li>
  <li>MNIST dataset (downloaded automatically when running the script)</li>
</ul>

<h3 id="running-the-script">Running the Script</h3>

<ol>
  <li>Clone this repository and navigate to the directory.</li>
  <li>Run the script using <code>python script.py</code> (assuming the script is named <code>script.py</code>).</li>
  <li>The script will download the MNIST dataset using the following code</li>
<pre><code>
  import torch 
  from torchvision import datasets
  train = datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor)
</code></pre>
  <li> save the model state to a file named <code>model_state.pt</code>.</li>
  <li>Once the training is complete, you can use the classifier to classify images by loading the saved model state and passing an image tensor to the <code>clf</code> object.</li>
</ol>

<h2 id="usage">Usage</h2>

<h3 id="training-the-classifier">Training the Classifier</h3>

<p>The script trains the classifier on the MNIST dataset for 10 epochs with a batch size of 32. You can adjust these hyperparameters by modifying the script.</p>

<h3 id="classifying-images">Classifying Images</h3>

<p>To classify an image, load the saved model state using <code>clf.load_state_dict(load(f))</code> and pass an image tensor to the <code>clf</code> object. The output will be the predicted class label.</p>

<pre><code>
img = Image.open('img_1.jpg')
img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
print(torch.argmax(clf(img_tensor)))
</code></pre>

<p>This will classify the image <code>img_1.jpg</code> and print the predicted class label.</p>
<h2 id="model-architecture">Model Architecture</h2>

<p>The classifier uses a convolutional neural network (CNN) with the following architecture:</p>

<ul>
  <li>Conv2d(1, 32, (3,3)) -> ReLU</li>
  <li>Conv2d(32, 64, (3,3)) -> ReLU</li>
  <li>Conv2d(64, 64, (3,3)) -> ReLU</li>
  <li>Flatten</li>
  <li>Linear(64*22*22, 10)</li>
</ul>

<p>The model is defined in the <code>ImageClassifier</code> class.</p>

<h2 id="license">License</h2>

<p>This code is licensed under the MIT License. See the LICENSE file for details.</p>

<h2 id="contributing">Contributing</h2>

<p>Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.</p>
