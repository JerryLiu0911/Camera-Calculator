# 📷✏️ Camera Calculator – Handwritten Equation Recognition App

***For a more detailed documentation see CamCalc_documentation.pdf***

This project is a mobile and desktop-compatible application designed to recognize handwritten mathematical expressions from images. It leverages a combination of computer vision, deep learning, and custom heuristics to extract, segment, classify, and interpret handwritten symbols.

---

## 🚀 Technologies Used

- **Python**
- **Kivy** – GUI framework for cross-platform apps
- **OpenCV** – Image processing
- **TensorFlow / Keras** – Deep learning (model training and inference)
- **TensorFlow Lite** – Lightweight model deployment on mobile
- **NumPy / Matplotlib / Sklearn** – Data manipulation and evaluation
- **Pathlib** – Filesystem navigation

---

## 🧠 Core Pipeline Overview

![image](https://user-images.githubusercontent.com/89786918/191934264-0c7d6577-6914-43a3-a429-99d6ac75864e.png)

---

### Image Processing 
Before the classification of numbers and symbols can be performed, the image needs to be preprocessed to allow uniform inputs into the neural network. The image will be denoised and reshaped using OpenCV, and segmented into its individual symbols. During this process, the segmented parts will be labelled to allow for reconstruction of the data after recognition, in which the order may be stored as an attribute to the symbols or numbers. 

Example output:


![image](https://github.com/user-attachments/assets/79656de4-c88d-454c-ae82-94b82e24834a)
![image](https://github.com/user-attachments/assets/18198f32-ec51-49ab-ad94-49f0bc27bc94)

---

### Neural Networks
For ease of designing the neural network and to maximize time in optimization, I will be converting the images obtained into NumPy arrays, as NumPy arrays use less memory, along with the ease of performing matrix transformations and other mathematical operations, all of which will be immensely useful for the development of this project. I have chosen to use Convolutional Neural Networks to perform optical character recognition due to their efficiency in image recognition as they require less preprocessing of the input images and the abundance of resources available. One of the biggest advantages of Convolutional neural networks is the ability to capture spatial dependencies; where a pixel’s value is influenced by the value of nearby pixels; by applying “filters” to the image.
 
 ![image](https://user-images.githubusercontent.com/89786918/192470116-8201c581-36ae-46dd-be23-c0273e8120fb.png)

▲An example of a CNN architecture used to classify clothing. I intend to implement a similar architecture. 

Convolutional Neural Networks (CNNs) use filters, often called kernels, which are typically small matrices such as 3×3 or 5×5. These kernels slide over the input image, multiplying their values by the corresponding pixel values in the image region they cover. This operation extracts local features by sampling pixels within the kernel’s region. For example, a 3×3 kernel processes a 3×3 pixel area at a time.

The movement of kernels across the image happens with a fixed stride, which represents the step size between each sampled region. The values inside the kernels are weights that are optimized during training through machine learning algorithms like backpropagation.

A typical CNN architecture consists of several types of layers working together:

- Convolutional Layers: Apply kernels to detect features such as edges, textures, and patterns.

- Max-Pooling Layers: Reduce spatial dimensions by grouping and summarizing features, improving computational efficiency and translation invariance.

- Fully Connected Layers: Interpret the extracted features to classify the input into specific categories.

Libraries such as TensorFlow will be used for the training and development of the neural network, whereas training data will be sourced from Kaggle, which will be cited in the References section.

---

## Supported Classes

The CNN model is trained to classify the following 16 classes:

| Label | Meaning    |
|-------|------------|
| 0     | Digit 0    |
| 1     | Digit 1    |
| 2     | Digit 2    |
| 3     | Digit 3    |
| 4     | Digit 4    |
| 5     | Digit 5    |
| 6     | Digit 6    |
| 7     | Digit 7    |
| 8     | Digit 8    |
| 9     | Digit 9    |
| add   | Plus (+)   |
| div   | Divide (÷) |
| eq    | Equals (=) |
| mul   | Multiply (*)|
| sub   | Minus (-)  |
| x     | Variable x |

---

## 📁 Module Breakdown

### main.py – GUI Application (Kivy)
- **Description**: Kivy-based GUI application that allows users to draw or upload handwritten math equations. It integrates with the segmentation and classification pipeline and displays recognized results.
- **Features**:
  - User-friendly interface for drawing or image upload.
  - Button controls for triggering recognition.
  - Result display for predicted equations.

### ImageSegmentationModule.py – Symbol Segmentation
- **Description**: Performs image preprocessing and character segmentation using OpenCV.
- **Key Functions**:
  - `segment(image_path, size, test)`: Segments the input image into individual character images resized to 50x50.
  - `segmentDataset()`: Processes entire datasets for training.
- **Techniques**:
  - Grayscale conversion, thresholding.
  - Contour detection and filtering based on aspect ratios.
  - Character sorting by horizontal position.
  - Grouping logic to combine symbols properly.

### Dataset Initialiser.py - CNN Training Data Loader
- **Description**: Loads and preprocesses image data from the directory into TensorFlow datasets.
- **Features**:
  - Directory-based loader with train/validation split.
  - Resizing and grayscale conversion to prepare 50x50 images.

### CNNTrain.py – CNN Architecture & Training
- **Description**: Defines, compiles, and trains a CNN model for 16-class classification of mathematical symbols.
- **Architecture**:
  - Rescaling input layer.
  - Two convolutional + max-pooling layers.
  - Flatten and three dense layers with dropout for regularization.
  - Output softmax layer with 16 classes.
- **Training**:
  - Adam optimizer, sparse categorical cross-entropy loss.
  - EarlyStopping callback for efficient training.
- **Output**:
  - Saved Keras model (`CNN` directory).
  - TensorFlow Lite model (`CNN.tflite`) for mobile deployment.

### CNN_test.py – Inference + Visualization
- **Description**: Runs inference on segmented images using the TensorFlow Lite model and evaluates the performance.
- **Evaluation Steps**:
  - Load and run TFLite model interpreter on segmented inputs.
  - Display predicted classes with confidence scores.
  - Visualize feature maps from the first convolutional layer.
  - Generate and display confusion matrix for validation set.
- **Outputs**:
  - Predicted symbol labels.
  - Feature map visualizations.
  - Confusion matrix plots.

---
## 📱Mobile Readiness

- The project uses **TensorFlow Lite** to convert the trained model into a lightweight format suitable for mobile deployment.
- The GUI is built with **Kivy**, which supports Android and iOS platforms.
- The segmentation and classification pipeline is optimized to run inference efficiently on mobile devices.

---

## 🛠 Future Improvements

- **Expand symbol set**: Include more math symbols such as parentheses, decimals, variables beyond 'x', and Greek letters.
- **Improve segmentation**: Enhance grouping logic to better handle complex equations with nested symbols.
- **Real-time recognition**: Add live camera input with real-time segmentation and prediction.
- **User feedback loop**: Allow manual correction of predictions to improve the dataset and model iteratively.
- **Model optimization**: Use pruning and quantization to further reduce model size and increase inference speed on mobile.
- **Multi-language support**: Expand to recognize handwritten text beyond math, such as letters or scientific notation.

---

## Sample Output
![image](https://github.com/user-attachments/assets/1fc2d31c-c74c-49dd-acfc-8c72f93950b3) ![image](https://github.com/user-attachments/assets/24206732-2d12-4203-8524-03be37cbe39f)
https://www.youtube.com/watch?v=m57YgKU0cWg


## Confusion Matrix
![image](https://github.com/user-attachments/assets/2b289299-28b4-4dfb-96ed-b966919b245e)

## References
- http://neuralnetworksanddeeplearning.com
- https://github.com/stfc-sciml/sciml-workshop
-https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7
https://en.wikipedia.org/wiki/Gaussian_blur
- https://en.wikipedia.org/wiki/Otsu%27s_method
- http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
- https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
-https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
https://en.wikipedia.org/wiki/Softmax_function
-https://www.tensorflow.org/api_docs/python/tf
-https://keras.io/api/
https://docs.python.org/3/library/itertools.html
-https://github.com/enggen/Deep-Learning-Coursera/tree/master/Convolutional%20Neural%20Networks
-https://kivy.org/doc/stable/
