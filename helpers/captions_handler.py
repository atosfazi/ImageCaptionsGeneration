import pandas as pd
from collections import defaultdict
import re


class CaptionsHandler:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.captions = defaultdict(dict)
        self.start_token = '<start>'
        self.end_token = '<end>'

    def load_captions(self):
        data = pd.read_csv(self.data_path, encoding='utf-8', sep='|')
        data.columns = data.columns.str.strip()
        data['comment'] = data['comment'].apply(self.__process_caption)
#        data[' comment'] = data[' comment'].apply(self.__add_start_and_end_tokens)
        data['image_name'] = data['image_name'].apply(lambda x: x.replace('.jpg', ''))
        return data

    def __add_start_and_end_tokens(self, text):
        return f'{self.start_token} {text} {self.end_token}'

    @staticmethod
    def __process_caption(text):
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'(["\'])(\s*)(.*?)(\s*)(\1)', r'\1\3\1', text)
        text = re.sub(r"\s*'\s*n\s*'\s*", r"'n'", text)
        text = re.sub(r"\s+'s\b", r"'s", text)
        text = re.sub(r'\s*\(\s*(.*?)\s*\)\s*', r'(\1)', text)
        return text.lstrip()
