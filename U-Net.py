import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from model import Unet

samI, samM = None

def resize(spec, target):
    spec = tf.image.resize(spec, (128,128), method="nearest")
    target = tf.image.resize(target, (128,128), method="nearest")
    return spec, target

def augment(spec, target):
    return spec, target

def normalize(spec, target):
    spec = tf.cast(spec, tf.float32) / 255.0
    target -= 1
    return spec, target

def load_image_train(datapoint):
    spec = datapoint["image"]
    target = datapoint["segmentation_mask"]
    spec, target = resize(spec, target)
    spec, target = augment(spec, target)
    spec, target = normalize(spec, target)

    return spec, target

def load_image_test(datapoint):
    spec = datapoint["image"]
    target = datapoint["segmentation_mask"]
    spec, target = resize(spec, target)
    spec, target = normalize(spec, target)

    return spec, target

def display(to_disp):
    plt.figure(figsize=(15,15))

    titles = ["Input", "True_Segmentation", "Predicted_Segmentation"]

    for i in range(len(to_disp)):
        plt.subplot(1, len(to_disp), i+1)
        plt.title(titles[i])
        plt.imshow(tf.keras.utils.array_to_img(to_disp[i]))
        plt.axis("off")
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([samI, samM, create_mask(model.predict(samI[tf.newaxis, ...]))])

if __name__ == "__main__":
    dataset, meta = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    training = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    testing = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    training_batches = training.cache().shuffle(1000).batch(64).repeat()
    training_batches = training_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_batches = testing.take(3000).batch(64)
    test_batches = testing.skip(3000).take(669).batch(64)
    
    sample_batch = next(iter(training_batches))
    randomI = np.random.choice(sample_batch[0].shape[0])

    samI, samM = sample_batch[0][randomI], sample_batch[1][randomI]
    display([samI, samM])

    model = Unet()
    model.summary()
    #keras.utils.plot_model(model)

    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = "sparse_categorical_crossentropy", metrics="accuracy")

    NUM_EPOCHS = 20
    TRAIN_LENGTH = meta.splits["train"].num_examples
    STEPS_PER_EPOCH = TRAIN_LENGTH // 64

    VAL_SUBSPLITS = 5
    TEST_LENGTH = meta.splits["test"].num_examples
    VALIDATION_STEPS = TEST_LENGTH // 64 // VAL_SUBSPLITS

    model_history = model.fit(training_batches, epochs = NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=test_batches)

    count = 0
    for i in test_batches:
        count += 1
    print("number of batches:", count)
