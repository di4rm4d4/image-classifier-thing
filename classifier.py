import time
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import logging
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#load flower dataset 
dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True)
training_set = dataset['train']
validation_set = dataset['validation']
test_set = dataset['test']
#idk why these splits work but yolo lmao
train_split = 70
test_val_split = 30

#number of examples in each split
train_count = dataset_info.splits['train'].num_examples
test_count = dataset_info.splits['test'].num_examples
total_count = train_count + test_count

# number of training, validation, and test examples
train_num = (total_count * train_split) // 100
val_num = (total_count * test_val_split) // 100
test_num = val_num

#printing stuff (i did this in jupyer notebooks initially so it was prettier)
print(f'Training set contains {train_num:,} images')
print(f'Validation set contains {val_num:,} images')
print(f'Test set contains {test_num:,} images')

# getting and printing the number of classes in the dataset
class_count = dataset_info.features['label'].num_classes
print(f'The dataset has {class_count:,} classes')

#printing shape and label of 3 images for ref
for img, lbl in training_set.take(3):
    img_array = img.numpy()
    lbl_array = lbl.numpy()
    
    print('Shape of image: ', img_array.shape)
    print('Label of image: ', lbl_array)

#plot image from training set for the funnies
for img, lbl in training_set.take(1):
    img = img.numpy().squeeze()
    lbl = lbl.numpy()

plt.imshow(img)
plt.title(lbl)

#label mapping

with open('label_map.json', 'r') as f:
    class_names = json.load(f)


#actually label this damn image
for img, lbl in training_set.take(1):
    img_arr = img.numpy()
    lbl_arr = lbl.numpy()

plt.imshow(img_arr)
plt.title(class_names[str(lbl_arr)])

#PIPELINEEEEE LETSGOOOO

img_size = 224
b_size = 60

def format_data(img, lbl):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (img_size, img_size))
    img /= 255.0
    return img, lbl

train_batches = (training_set
                 .cache()
                 .shuffle(train_num // 4)
                 .map(format_data)
                 .batch(b_size)
                 .prefetch(1))

test_batches = (test_set
                .map(format_data)
                .batch(b_size)
                .prefetch(1))

val_batches = (validation_set
               .map(format_data)
               .batch(b_size)
               .prefetch(1))

#build and train
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_layer = hub.KerasLayer(url, input_shape=(img_size, img_size, 3))
feature_layer.trainable = False

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(102, activation='softmax')
])

model.summary()

print('Is there a GPU Available:', tf.config.list_physical_devices('GPU'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

epochs = 12

history = model.fit(
    train_batches,
    epochs=epochs,
    validation_data=val_batches,
    callbacks=[early_stop]
)

#plot the loss and accuracy levels for reference bc we are NOT betas
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

epoch_range = range(epochs)

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epoch_range, train_acc, label='Training Accuracy')
plt.plot(epoch_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoch_range, train_loss, label='Training Loss')
plt.plot(epoch_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

#save model as a Keras model and load
model.save('saved_model.h5')
reloaded_model = tf.keras.models.load_model('saved_model.h5', custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

def preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, (img_size, img_size))
    img /= 255.0
    img = img.numpy()
    return img

#omg i had a seizure getting PIL to work but yeah this plots the original image next to the processed one bc we want everything to be the same size for training
from PIL import Image

image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = preprocess_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()

#predictionnn
def predict(image_path, model, k):
    
    with Image.open(image_path) as img:
        img_array = np.array(img)
        
    processed_img = preprocess_image(img_array)
    batch_img = np.expand_dims(processed_img, axis=0)
    prediction_probs = model.predict(batch_img)
    
    top_k_probabilities = tf.math.top_k(prediction_probs, k=top_k)
    values = top_k_probabilities.values.numpy()
    indices = top_k_probabilities.indices.numpy()
    
    return values, indices, processed_img

#plot the input images and the top 5 classes + probabilities

image_files = [
    'cautleya_spicata.jpg', 
    'hard-leaved_pocket_orchid.jpg', 
    'orange_dahlia.jpg', 
    'wild_pansy.jpg'
]


top_k = 5

for image_filename in image_files:
    try:
        image_path = f'./test_images/{image_filename}'
        
        probabilities, class_indices, processed_image = predict(image_path, reloaded_model, top_k)
        
    except IOError:
        continue
    
    class_names_list = [class_names[str(index + 1)] for index in class_indices[0]]

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)
    

    ax1.imshow(processed_image)
    ax1.axis('off')
    ax1.set_title(class_names_list[0])
    
    ax2.barh(np.arange(top_k), probabilities[0])
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(class_names_list, size='small')
    ax2.set_aspect(0.1)
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Class Probability')
    
    plt.tight_layout()
    plt.show()
