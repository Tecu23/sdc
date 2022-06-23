import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import  img_to_array, load_img
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.models import load_model

"""
  Metoda pentru incarcarea imaginilor pentru antrenament si testare
    IN: Folderul cu date
    OUT: Imaginile de format 30x30x3 si categoriile
"""
def load_data(data_dir):
    
    images = list()
    labels = list()

    # 4 categorii de semne (30, 50, stop, nu e semn)
    for category in range(4):
        categories = os.path.join(data_dir, str(category))
        for img in os.listdir(categories):
            img = load_img(os.path.join(categories, img), target_size=(30, 30))
            image = img_to_array(img)
            images.append(image)
            labels.append(category)
    
    return images, labels

"""
  Metoda pentru crearea modelului
    IN: Dimensiunea imaginiilor si numarul de output
    OUT: Modelul creat
"""

def create_model(IMG_HEIGHT, IMG_WIDTH, output_classes):
  #============================================= Modelul ==================================================
  #========================================================================================================
  model = Sequential()
  # Primul nivel Convolutional
  model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(rate=0.25))
  # Al doilea nivel convolutional
  model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(rate=0.25))
  # Aplatizarea retelei pentru categorizare
  model.add(Flatten())
  model.add(Dense(units=32, activation='relu'))
  model.add(Dense(output_classes, activation='softmax'))
  model.summary()
  #========================================================================================================

  return model

"""
  Metoda pentru antrenarea retelei neuronale
    IN: Folderul de train, dimensiunile imaginilor, epochs, modelul
    OUT: Salvarea modelului intr-un fisier
"""
def train_SignsModel(data_dir,IMG_HEIGHT = 30,IMG_WIDTH = 30,EPOCHS = 30, save_model = True,saved_model = "data/model.h5"):
    
    train_path = data_dir + '/train'
    output_classes = 4

    # 1. Incarcam datele pentru antrenament
    images, labels = load_data(train_path)
    labels = to_categorical(labels)

    # 2. Impartim setul de date in train si test
    x_train, x_test, y_train, y_test = train_test_split(np.array(images), labels, test_size=0.4)

    # 3. Cream modelul
    model = create_model(IMG_HEIGHT, IMG_WIDTH, output_classes)

    # 4. Compilare model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    # 5. Potrivim modelul
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=EPOCHS, steps_per_epoch=60)

    print(x_test.shape)
    print(y_test.shape)
    print(y_test)

    # 6. Evaluam si afisam modelul
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test set accuracy: ', accuracy * 100)

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    #========================================= Saving Model =================================================
    #========================================================================================================
    # save model and architecture to single file
    if save_model:
        model.save(saved_model)
        print("Saved model to disk")
    #========================================================================================================


"""
  Metoda pentru evaluarea modelului pentru o imagine anume
    IN: Modelul, imaginea si eticheta imaginii
    OUT: Evaluarea imaginii
"""
def EvaluateModelOnImage(model_path,image_path,image_label):
    model = load_model(model_path)

    output = []

    image = load_img(image_path, target_size=(30, 30))

    output.append(np.array(image))

    X = np.array(image).reshape(1,30,30,3)

    if image_label == 0:
        Y = np.array([[1,0,0,0]])
    elif image_label == 1:
        Y = np.array([[0,1,0,0]])
    elif image_label == 2:
        Y = np.array([[0,0,1,0]])
    else:
        Y = np.array([[0,0,0,1]])
        
    print(X.shape)
    print(Y.shape)
    # evaluate the model
    score = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))