from tensorflow.data import Dataset
import tensorflow as tf

from helpers.captions_handler import CaptionsHandler
from helpers.tokenizer import Tokenizer
from helpers.data_generator import generate_batches
from helpers.model import build_model


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
            (tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),
            tf.TensorSpec(shape=(None, tokenizer.max_seq_length-1), dtype=tf.int32)),
            tf.TensorSpec(shape=(None, tokenizer.max_seq_length-1), dtype=tf.int32),
        )
    )
    dataset = dataset.shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    val_size = int(0.15 * dataset_size)

    val_dataset = dataset.take(val_size)
    train_dataset = dataset.skip(val_size)

    model = build_model(max_seq_len=tokenizer.max_seq_length-1)
    print(model.summary())

    model.fit(train_dataset, validation_data=val_dataset, epochs=3)


if __name__ == '__main__':
    main()