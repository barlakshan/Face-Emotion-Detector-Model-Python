# Emotion Detection using CNN

This project aims to develop a Convolutional Neural Network (CNN) for emotion detection from grayscale images. The model is trained on a dataset of images categorized by different emotions.

## Features

- **Image Preprocessing**: Load and preprocess images from a dataset.
- **Data Augmentation**: Augment training data to improve model generalization.
- **Model Training**: Train a CNN model to classify emotions.
- **Model Evaluation**: Evaluate the model's performance using classification report and confusion matrix.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/emotion-detection-cnn.git
    ```

2. Install the required libraries:
    ```sh
    pip install numpy matplotlib opencv-python scikit-learn seaborn tensorflow
    ```

3. Mount Google Drive (if using Google Colab):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Usage

1. **Data Path Setup**: Set the path to the directory containing images:
    ```python
    data_path = '/content/drive/My Drive/Colab Notebooks/Nature inspired/Group_Mini/Emotion Detection/train'
    ```

2. **Data Preprocessing and Exploration**:
    ```python
    import numpy as np
    import os
    import cv2

    images = []
    labels = []
    categories = os.listdir(data_path)

    for category in categories:
        category_path = os.path.join(data_path, category)
        label = categories.index(category)
        for img_filename in os.listdir(category_path):
            try:
                img = cv2.imread(os.path.join(category_path, img_filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))
                img = img / 255.0
                images.append(img)
                labels.append(label)
            except Exception as e:
                print("Error loading image:", e)

    images = np.array(images)
    labels = np.array(labels)
    ```

3. **Train-Test Split**:
    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    ```

4. **Data Augmentation**:
    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    ```

5. **Model Architecture**:
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(categories), activation='softmax')
    ])
    ```

6. **Model Compilation and Training**:
    ```python
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=50, validation_data=(X_test, y_test), verbose=1)
    ```

7. **Model Evaluation**:
    ```python
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_classes))

    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    ```

## Contributors

This project is a group assignment completed by the following contributors:

- **Contributor 1** - [Eranda-Uditha](https://github.com/Eranda-Uditha)
- **Contributor 2** - [SSandaruwanSrimal](https://github.com/SSandaruwanSrimal)
- **Contributor 3** - [Eranga0619](https://github.com/Eranga0619)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- TensorFlow and Keras for providing the deep learning framework.
- OpenCV for image processing utilities.
- The data source for providing the images used for training and testing.

