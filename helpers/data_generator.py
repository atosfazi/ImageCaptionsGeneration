import os
import numpy as np
from typing import Dict, List, Union, DefaultDict
import random


def generate_batches(features_dir: str,
                     captions: DefaultDict[str, Dict[str, List[Union[str, int]]]],
                     batch_size: int):
    image_filenames = list(captions.keys())
    samples_nb = sum([len(image['captions']) for image in captions.values()])

    while True:
        for offset in range(0, samples_nb, batch_size):
            batch_filenames = image_filenames[offset:offset + batch_size]

            batch_image_inputs = []
            batch_text_inputs = []
            batch_labels = []

            for filename in batch_filenames:
                img_input = np.load(os.path.join(features_dir, f'{filename}.npy'))
                img_input = np.squeeze(img_input, axis=0)
                text_inputs = captions[filename]['inputs']
                targets = captions[filename]['outputs']
                idx = random.randint(0, len(text_inputs) - 1)
                batch_image_inputs.append(img_input)
                batch_text_inputs.append(text_inputs[idx])
                batch_labels.append(np.array(targets[idx], dtype=np.int32))

            if len(batch_text_inputs) > 0:
                yield (np.array(batch_image_inputs), np.array(batch_text_inputs)), np.array(batch_labels)
            else:
                print("Warning: Empty batch, skipping.")
