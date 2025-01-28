from tensorflow.data import Dataset
import tensorflow as tf
import numpy as np
import os

from helpers.captions_handler import CaptionsHandler
from helpers.tokenizer import Tokenizer
from helpers.data_generator import generate_batches


def main():
    path = r'dataset/results.csv'
    image_features_dir = 'dataset/image_features'
    captions_handler = CaptionsHandler(data_path=path)
    captions = captions_handler.load_captions()

    tokenizer = Tokenizer()
    captions = tokenizer.tokenize_sequences(captions)

    dataset = Dataset.from_generator(
        lambda: generate_batches(image_features_dir, captions, 32),
        output_signature=(
            tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 88), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 88), dtype=tf.int32),
        )
    )
    dataset = dataset.shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)


if __name__ == '__main__':
    main()