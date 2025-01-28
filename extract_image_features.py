import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
from tqdm import tqdm


# constrains
image_dataset_path = r'dataset/flickr30k_images'
save_dir = 'dataset/image_features'
os.makedirs(save_dir, exist_ok=True)


def preprocess_image(image_path, target_size=(224, 224)):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


if __name__ == '__main__':
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.output)
    for file in tqdm(os.listdir(image_dataset_path)):
        file_path = os.path.join(image_dataset_path, file)
        try:
            image = preprocess_image(file_path)
            features = feature_extractor(image)
            save_path = os.path.join(save_dir, file.replace('.jpg', '.npy'))
            np.save(save_path, features)
        except Exception as e:
            print(f'Error processing image {file_path}: {e}')

