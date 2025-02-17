import os
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.backend import image_data_format
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D, concatenate, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.utils import get_file
from keras.applications.inception_v3 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from tensorflow.keras.utils import plot_model

WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def inceptionModuleA(x,unique_filters,concat_axis,name=None):
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, unique_filters, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=concat_axis,
        name='mixed'+name)
    return x

def inceptionModuleB(x,unique_filters,concat_axis,name=None):
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, unique_filters, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, unique_filters, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, unique_filters, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, unique_filters, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, unique_filters, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, unique_filters, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=concat_axis,
        name='mixed' + name)
    return x

def inceptionModuleC(x,concat_axis,name=None):
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = concatenate(
        [branch3x3_1, branch3x3_2],
        axis=concat_axis,
        name='mixed9_' + name)

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = concatenate(
        [branch3x3dbl_1, branch3x3dbl_2], axis=concat_axis)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        axis=concat_axis,
        name='mixed' + str(9 + int(name)))
    return x

def auxiliary_classifier(x,num_classes):
    x1 = AveragePooling2D((5,5),strides=(3,3))(x)
    x1 = conv2d_bn(x1, 128,1,1)
    x1 = Dense(num_classes, activation='softmax',name='auxillary_classifier')(x1)
    return x1

def load_weights(model,weights,include_top):
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    
    return model

def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=75,
        data_format=image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = inceptionModuleA(x,32,channel_axis,"0")
    x = inceptionModuleA(x,64,channel_axis,"1")
    x = inceptionModuleA(x,64,channel_axis,"2")

    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')
    x = inceptionModuleB(x,128,channel_axis,"4")

    x = inceptionModuleB(x,160,channel_axis,"5")
    x = inceptionModuleB(x,160,channel_axis,"6")

    x = inceptionModuleB(x,192,channel_axis,"7")


    x1 = auxiliary_classifier(x,classes)

    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    x = inceptionModuleC(x,channel_axis,"1")
    x = inceptionModuleC(x,channel_axis,"2")


    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    if input_tensor is not None:
        inputs = tf.keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    
    model = Model(inputs, x, name='inception_v3')

    model = load_weights(model,weights,include_top)
    print("Inception v3 model created")

    return model

model = InceptionV3()

model.summary()

plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

local_image_path = "berta3.jpg"  

image = cv2.imread(local_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

def resize_img(img):
    img = cv2.resize(img, (299, 299))
    img = img.reshape(1, 299, 299, 3)
    img = preprocess_input(img)
    return img

img = resize_img(image)

result = model.predict(img)
pred = decode_predictions(result, top=5)

def show_image(img, title=None):
    plt.imshow(img)
    if title:
        plt.title(title.upper(), color='g')
    plt.show()

show_image(image, pred[0][0][1])  

DATASET_PATH = "dataset" 
IMG_SIZE = (299, 299)  
BATCH_SIZE = 16  
EPOCHS = 10  

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2  
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

base_model = InceptionV3(weights='imagenet', include_top=False) 

for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)  

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")

plt.show()

model.save("dog_cat_classifier.h5")

def predict_image(image_path, model_path="dog_cat_classifier.h5"):
    model = tf.keras.models.load_model(model_path)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    
    labels = {v: k for k, v in train_generator.class_indices.items()}
    predicted_label = labels[class_idx]
    
    plt.imshow(cv2.imread(image_path)[:, :, ::-1])
    plt.title(f"Prediction: {predicted_label}")
    plt.axis("off")
    plt.show()

test_image_path = "berta3.jpg" 
predict_image(test_image_path)