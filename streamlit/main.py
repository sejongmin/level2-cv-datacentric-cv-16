import json
import pandas as pd
import streamlit as st
from pathlib import Path

import visualize_ocr

class_colors = {
    'paragraph': (255, 241, 168),
    'word': (208, 56, 78), 
    'chars': (238, 100, 69)
}

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

@st.cache_data
def load_train_df(json_path='./code/data/chinese_receipt/ufo/train.json'):
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

    bbox_df = pd.DataFrame()
    bbox_df['word_ids'] = train_df['word_ids'].values
    bbox_df['transcriptions'] = train_df['transcriptions'].values
    bbox_df['x1'] = train_df['x1'].values
    bbox_df['y1'] = train_df['y1'].values
    bbox_df['x2'] = train_df['x2'].values
    bbox_df['y2'] = train_df['y2'].values
    bbox_df['x3'] = train_df['x3'].values
    bbox_df['y3'] = train_df['y3'].values
    bbox_df['x4'] = train_df['x4'].values
    bbox_df['y4'] = train_df['y4'].values

    return train_df, bbox_df
        
if "train_df" not in st.session_state:
    st.session_state.train_df = pd.DataFrame()
if "bbox_df" not in st.session_state:
    st.session_state.train_df = pd.DataFrame()

st.sidebar.success("CV16 오늘도 화이팅하조!")
with st.sidebar.form(key="json_form"):
    json_path = st.text_input("json file path")
    submit_button = st.form_submit_button("OK")
    if submit_button:
        try:
            st.session_state.train_df, st.session_state.bbox_df = load_train_df(json_path)
            st.sidebar.success("json file load successed :)")
        except Exception:
            st.session_state.train_df, st.session_state.bbox_df = load_train_df()
            st.sidebar.error("json file load failed :(")

st.markdown("<h2 style='text-align: center;'>OCR</h2>", unsafe_allow_html=True)
option = st.sidebar.radio("option", ["train images", "test images"])
if option == "train images" and not st.session_state.train_df.empty:
    st.session_state.image_ids = [img_id for img_id in st.session_state.train_df.groupby("image_id")["image_id"].first().tolist()]
    image_index = st.sidebar.slider('Select image id', 0, len(st.session_state.image_ids), 0)
    image_index_input = st.sidebar.number_input('Enter image count', min_value=0, max_value=len(st.session_state.image_ids), value=image_index, step=1)
    if image_index != image_index_input:
        image_index = image_index_input
    image_id = st.session_state.image_ids[image_index]
    visualize_ocr.show(st.session_state.train_df, image_id, json_path, class_colors)
elif option == "test images" and not st.session_state.test_df.empty:
    st.session_state.image_ids = [img_id for img_id in st.session_state.test_df.groupby("image_id")["image_id"].first().tolist()]
else:
    pass
