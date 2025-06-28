 ##Pepper Disease Classification
A deep learning project to classify images of pepper plants into healthy or diseased categories using TensorFlow and image processing techniques.

 ## Technologies Used

Technology
Purpose
Python
Programming language used for all development.
TensorFlow
Deep learning framework used for model creation, training, and evaluation.
Keras
High-level API of TensorFlow to simplify neural network construction.
Matplotlib
Used to visualize training results and model accuracy.
PlantVillage Dataset
A dataset directory with categorized images of pepper leaves.


 ## Dataset Information
Source Directory: PlantVillage/


Classes: 2 (Healthy and Diseased)


Images: 2475 total


Preprocessing: Images resized to 256x256 pixels and batched for training



 

 ## Workflow
1. Data Loading
tf.keras.preprocessing.image_dataset_from_directory(...)

Automatically labels folders as classes.


Loads, shuffles, and resizes image data.


2. Data Augmentation
Implemented via Sequential preprocessing layers:
Random flip (horizontal)


Random rotation


Random zoom


3. Model Architecture
models.Sequential([
    layers.InputLayer(input_shape=(256, 256, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(...),
    layers.MaxPooling2D(...),
    ...
])

Standard Convolutional Neural Network (CNN) with:


Multiple Conv2D layers


MaxPooling layers


Dropout for regularization


Dense output layer with softmax activation


4. Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

5. Training
model.fit(train_ds, validation_data=val_ds, epochs=10)

The model is trained for multiple epochs.


Validation data helps monitor overfitting.


6. Evaluation and Visualization
Accuracy and loss are plotted using matplotlib.


Final accuracy metrics are printed and visualized.



 ## Results
Training and validation accuracy improve over epochs.


The model can distinguish between healthy and diseased leaves effectively.






 ## How to Run
Make sure you have the dataset organized in folders like:
PlantVillage/
├── Healthy/
└── Diseased/

Install dependencies:
pip install tensorflow matplotlib

Run the notebook in Jupyter or VSCode.

