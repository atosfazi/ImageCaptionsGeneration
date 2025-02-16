import pandas as pd
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.max_seq_length = 0
        self.max_vocab_size = 5000
        self.word_index = {**self.tokenizer.get_vocab(), "<OOV>": 1}

    def tokenize_sequences(self, captions_data: pd.DataFrame):
        captions_data['tokenized'] = captions_data['comment'].apply(lambda x:
                                                                self.tokenizer.encode(x, add_special_tokens=True))
        self.max_seq_length = captions_data['tokenized'].apply(len).max()
        captions_data['tokenized'] = captions_data['tokenized'].apply(lambda x:
                                                                      pad_sequences([x],
                                                                                    maxlen=self.max_seq_length,
                                                                                    padding="post")[0])
        # captions_data['inputs'] = [list(row[:-1]) for row in captions_data['tokenized'].values]
        # captions_data['targets'] = [list(row[1:]) for row in captions_data['tokenized'].values]
        #
        captions_data['inputs'] = captions_data['tokenized'].apply(lambda seq: [self.token_to_index(idx) for idx in seq[:-1]])
        captions_data['targets'] = captions_data['tokenized'].apply(lambda seq: [self.token_to_index(idx) for idx in seq[1:]])

        return captions_data.groupby('image_name').apply(lambda group: {
            'captions': group['comment'].tolist(),
            'inputs': group['inputs'].tolist(),
            'outputs': group['targets'].tolist()}).to_dict()

    def token_to_index(self, token_id):
        if token_id >= self.max_vocab_size:
            return self.word_index["<OOV>"]
        return token_id
