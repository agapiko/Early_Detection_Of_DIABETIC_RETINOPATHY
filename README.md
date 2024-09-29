## Pioneering Diabetic Retinopathy Detection


 
<br>
<div style="text-align: center;">
  <figure>
    <img src="./Logo.png" alt="EagleEye" style="width:35%">
    <br>
    <figcaption>EagleEye: The Future in Diabetic Retinopathy Proactive Treatment</figcaption>
  </figure>
</div>
<br>


### Project Overview

At the forefront of medical diagnostics innovation, EagleEye is pioneering the application of advanced deep learning techniques to revolutionize the early detection of diabetic retinopathy (DR). Our project harnesses the power of the DenseNet-121 model, a cutting-edge convolutional neural network renowned for its exceptional performance in handling complex image data. By leveraging a comprehensive dataset of fundus camera images meticulously labeled for various stages of DR, our model integrates sophisticated preprocessing techniques to enhance image quality and optimize diagnostic accuracy. This groundbreaking initiative is poised to transform the landscape of healthcare delivery, equipping ophthalmologists and healthcare providers with a potent tool in the quest to combat diabetic blindness and improve patient outcomes.

---

### About Us

**EagleEye** is a pioneering healthcare technology company that aims to revolutionize the diagnosis and treatment of diabetic retinopathy with AI-powered image analysis. In the complex landscape of today's healthcare, the company addresses the vital need for accurate, efficient, and swift diagnosis. EagleEye's specialization lies in detecting diabetic retinopathy, and it offers an AI-powered diagnostic system tailored for healthcare professionals. This initiative is particularly significant for ophthalmology, a field often challenged by the timeliness and precision of retinal image interpretation.

**Our Mission and Vision**
EagleEye's mission is to provide a more accurate, efficient, and accessible diagnostic solution for healthcare providers and patients alike. Our vision is to make advanced ophthalmic diagnostics accessible to all healthcare facilities, from major hospitals to local clinics, and to equip healthcare providers worldwide with the tools they need to deliver fast, accurate, and early diagnoses of diabetic retinopathy. We aim to reduce the global burden of vision loss caused by diabetic retinopathy and to improve patient outcomes through timely interventions.

**Our Technology**
Our system integrates cutting-edge machine learning algorithms to provide a detailed analysis of retinal images. By using the DenseNet-121 model, our system enhances feature extraction and utilizes depth to accurately assess and classify the stages of diabetic retinopathy. Our platform empowers patients to take a more active role in their eye health by providing them with personalized information about their diabetic retinopathy risk and severity, enabling them to work more effectively with their healthcare providers to manage their condition.


**Business Strategy**
From a business perspective, EagleEye strategically caters to various sectors, including healthcare, education, and the workplace. Our objectives encompass diagnostic support, telemedicine capabilities, workplace injury prevention, and patient health monitoring. What sets us apart is our commitment to simplicity, model accuracy, and dedicated customer support. We distinguish ourselves from more complex competitors through our adherence to the Blue Ocean Strategy, an approach that prioritizes innovation and differentiation.

**Objectives and Strategies**
To achieve our business goals, we have set the following objectives:

*   <u>Improve Diagnostic Accuracy:</u> Achieve high accuracy in detecting diabetic 
retinopathy and its stages using image analysis powered by AI.
Increase Early Detection Rates: Increase the number of patients diagnosed with diabetic retinopathy at an early stage, enabling timely interventions and preventing vision loss.
*   <u>Reduce Healthcare Costs:</u> Reduce healthcare costs associated with diabetic retinopathy by minimizing the need for invasive treatments and hospitalizations.
*   <u>Enhance Patient Experience:</u> Provide a seamless and user-friendly experience for patients, enabling them to easily access and manage their eye health.
Our strategies for achieving our objectives include integrating our system with Electronic Health Records (EHRs) to enable seamless data exchange and streamline clinical workflows. We also plan to enable remote image analysis and reporting, which will facilitate telemedicine consultations and reduce the need for in-person visits.


**Technical Implementations**

Our technical approach focuses on refining model performance through techniques such as data augmentation, metric selection, and training callbacks. We employ the DenseNet-121 model, which has demonstrated superior performance in detecting diabetic retinopathy.

DenseNet-121 Architecture

The DenseNet-121 model is a variant of the DenseNet architecture, optimized for image classification tasks. It features a reduced number of parameters, achieved through a smaller growth rate and fewer dense blocks. This reduction in parameters leads to improved computational efficiency, memory efficiency, and training speed.

**Implementation Details**

We implemented the DenseNet-121 model using the following techniques:

*   Loaded and preprocessed the dataset, resizing images to 224x224 pixels
Applied data augmentation to enhance model performance
*   Employed an ordinal classification approach to capture the progressive nature of diabetic retinopathy
*   Defined custom metrics, including accuracy and Kappa score, to evaluate model performance
*   Modified the pre-trained model by excluding top fully connected layers and adding custom layers
*   Added an output layer with sigmoid activation for multi-label classification
*   Compiled the model using binary cross-entropy loss and Adam optimizer
*   Trained the model using the Keras fit method with early stopping and learning rate adjustment callbacks

**Model Evaluation**

We evaluated the model's performance using learning curves to visualize key metrics such as accuracy, loss, and Quadratic Weighted Kappa score across training epochs. The model achieved a validation accuracy of 0.82 and a QWK score of 0.90, indicating a high level of agreement and accuracy. The early stopping mechanism prevented overfitting, and the model demonstrated good generalization capabilities.


---

### Technical Installation - Python Environment Specifications
#### Libraries and Versions
The main libraries utilized are the following:
- **NumPy**: Version 1.26.4
- **Pandas**: Version 2.1.4
- **OpenCV**: Version 4.10.0
- **TensorFlow**: Version 2.17.0
- **Matplotlib**: Version 3.7.1
- **Seaborn**: Version 0.13.1
- **Tqdm**: Version 4.66.5
- **PrettyTable**: Version 3.11.0
- **Scikit-learn**: Version 1.5.2
- **Pillow**: Version 10.4.0
- **JSON**: Version 2.0.9

#### Detailed Specifications:
- **NumPy**: Version 1.26.4, essential for numerical operations, complements Pandas for data transformation and manipulation.
- **Pandas**: Version 2.1.4, provides robust capabilities for data manipulation and analysis.
- **OpenCV**: Version 4.10.0, essential for computer vision tasks, including image processing.
- **TensorFlow**: Version 2.17.0, forms the backbone for deep learning tasks, offering a flexible platform for building, training, and deploying machine learning models.
- **Matplotlib**: Version 3.7.1, offering a wide array of functionalities for data visualization.
- **Seaborn**: Version 0.13.1, extends Matplotlib for advanced visualizations.
- **Tqdm**: Version 4.66.5, helps in visualizing progress bars for loops and data processing tasks.
- **PrettyTable**: Version 3.11.0, useful for displaying tabular data in an easily readable format.
- **Scikit-learn**: Version 1.5.2, a comprehensive library for machine learning tasks such as data splitting, model training, and evaluation.
- **Pillow**: Version 10.4.0, used for image processing tasks.
- **JSON**: Version 2.0.9, used for encoding and decoding JSON data.
 
#### Important Notes
*WARNING:* You may encounter the following message when saving models in this repository:
*WARNING:* absl: You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
Rest assured, this is not a problem and will not affect the functionality of the repository.
 
The current analysis in this repository was conducted using Python 3 on Google Colab.

---

### Deliverables
1. [Documentation](#documentation)
2. [README](#readme)
3. [Dataset](#dataset)
4. [Notebooks](#notebooks)
5. [Requirements](#requirements)
6. [Model Weights](#model-weights)
 
### Documentation
Retinopathy_detection.pdf file are available for an in-depth understanding of the project.

### README
The README.md file offers an overview and provides navigation guidance for this repository.

### Datasets
The dataset.txt file contains the dataset details used for training and validation of the models.

### Notebooks
This repository contains three types of Jupyter Notebooks:
- CNN.ipynb: Outlines the architecture and training procedures for the CNN model.
- InceptionNetV3.ipynb: Provides the implementation details for the InceptionNetV3 model
- DenseNet121.ipynb: Focuses on the DenseNet121 model and its specifics for medical image classification.

### Requirements
The requirements.txt file lists all the dependencies needed to run the code in this repository.

### Model Weights
The weights.txt file provides links to access files containing weights for the best epochs of our models.

---

### Contributors

- [x] [Agapi Koutsogianni](https://github.com/agapiko)
- [x] [Anna Merlou](https://github.com/MerlouAnna)
- [x] [Athina Ksanalatou](https://github.com/athinaksan)
- [x] [Stavros Ziaragkalis](https://github.com/FoteiniNefeli)


