import os
import numpy as np
from typing import DefaultDict, Dict, List, Union
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
                text_inputs = captions[filename]['inputs']
                targets = captions[filename]['targets']
                idx = random.randint(0, len(text_inputs) - 1)

                batch_image_inputs.append(img_input)
                batch_text_inputs.append(text_inputs[idx])
                batch_labels.append(targets[idx])

            yield np.array(batch_image_inputs), np.array(batch_text_inputs), np.array(batch_labels)
