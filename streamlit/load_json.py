import json
import pandas as pd
import streamlit as st
from pathlib import Path

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

@st.cache_data
def load_df(json_path='./code/data/chinese_receipt/ufo/train.json'):
    data = {}
    data['images'] = {}
    json_data = read_json(json_path)
    images = list(json_data['images'].items())
    data['images'].update(dict(images))

    train_df = pd.DataFrame()
    image_ids = []
    word_ids = []
    image_heights = []
    image_widths = []
    transcriptions = []
    x1, x2, x3, x4 = [], [], [], []
    y1, y2, y3, y4 = [], [], [], []

    for image_key, image_value in list(data["images"].items()):
        file_name = image_key
        word_ann = image_value['words']
        for word in word_ann.values():
            image_ids.append(file_name)
            word_ids.append(word)
            image_heights.append(image_value['img_h'])
            image_widths.append(image_value['img_w'])
            transcriptions.append(word['transcription'])
            if not word['points']:
                x1.append("")
                y1.append("")
                x2.append("")
                y2.append("")
                x3.append("")
                y3.append("")
                x4.append("")
                y4.append("")
                continue
            x1.append(float(word['points'][0][0]))
            y1.append(float(word['points'][0][1]))
            x2.append(float(word['points'][1][0]))
            y2.append(float(word['points'][1][1]))
            x3.append(float(word['points'][2][0]))
            y3.append(float(word['points'][2][1]))
            x4.append(float(word['points'][3][0]))
            y4.append(float(word['points'][3][1]))

    train_df['image_id'] = image_ids
    train_df['word_ids'] = word_ids
    train_df['img_w'] = image_widths
    train_df['img_h'] = image_heights
    train_df['transcriptions'] = transcriptions
    train_df['x1'] = x1
    train_df['y1'] = y1
    train_df['x2'] = x2
    train_df['y2'] = y2
    train_df['x3'] = x3
    train_df['y3'] = y3
    train_df['x4'] = x4
    train_df['y4'] = y4

    return train_df