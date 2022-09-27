# Camera Calculator

As a student, when tasked with completing math assignments and homework, seeking help from calculators is a common habit of many. However, as we progress through education the mathematical equations we face grow in complexity, where pre-installed calculators on phones and commonplace household calculators often offer little assistance. Online calculators although providing more functionality, transcribing mathematical expressions to text can often be a tedious task; some online calculators use the LaTeX language to express mathematical notation and symbols, but this requires users to learn its keywords. Other online calculators simply let users select symbols from a keyboard, which can also be time-consuming when dealing with longer expressions.

To improve the efficiency of inputting mathematical expressions and calculating them, I intend to create a mobile phone app which utilizes the camera functionality to recognize handwritten mathematical expressions and perform the desired calculations. I aim to be able to solve simple, one-variable linear equations and expressions, and gradually implement more functionality as the project progresses. The app will be designed for those who are looking for a quicker way to enter equations into their calculators.


## Modelling
![image](https://user-images.githubusercontent.com/89786918/191934264-0c7d6577-6914-43a3-a429-99d6ac75864e.png)

## Image Processing
Before the classification of numbers and symbols can be performed, the image needs to be preprocessed to allow uniform inputs into the neural network. The image will be denoised and reshaped using OpenCV, and segmented into its individual symbols. During this process, the segmented parts will be labelled to allow for reconstruction of the data after recognition, in which the order may be stored as an attribute to the symbols or numbers. 

## Neural Networks
For ease of designing the neural network and to maximize time in optimization, I will be converting the images obtained into NumPy arrays, as NumPy arrays use less memory, along with the ease of performing matrix transformations and other mathematical operations, all of which will be immensely useful for the development of this project. I have chosen to use Convolutional Neural Networks to perform optical character recognition due to their efficiency in image recognition as they require less preprocessing of the input images and the abundance of resources available. One of the biggest advantages of Convolutional neural networks is the ability to capture spatial dependencies; where a pixel’s value is influenced by the value of nearby pixels; by applying “filters” to the image.
 
 ![image](https://user-images.githubusercontent.com/89786918/192470116-8201c581-36ae-46dd-be23-c0273e8120fb.png)

▲An example of a CNN architecture used to classify clothing. I intend to implement a similar architecture. 

These filters, often referred to as "kernels," are typically 3 x 3 or 5 x 5 matrices that are multiplied by the image's pixel values to identify desired characteristics by sampling the same amount of pixels as their size (aka 3 x 3 kernels sample pixels in a 3 x 3 region); repeating the process across the image with a fixed “stride”, which can be thought of the distance between each sampled region. The values within the kernels are the weights which may be optimized through training where machine learning algorithms such as backpropagation may be used. The application of the kernels will be done through layers, each layer in the CNN having a different purpose. In the typical CNN, the architecture consists of a combination of convolutional layers to filter the image, a max-pooling layer to standardize and group convoluted features for ease of computing, and a fully connected layer which is responsible for classifying the collected features into their respective classes.
Libraries such as TensorFlow will be used for the training and development of the neural network, whereas training data will be sourced from Kaggle, which will be cited in the References section. 
