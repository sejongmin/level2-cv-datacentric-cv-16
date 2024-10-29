import cv2
import numpy as np
import streamlit as st

def draw_bbox(image, coordinates, color=(0, 255, 0), thickness=2):
    points = np.array(coordinates).reshape(-1, 2).astype(int)
    for i in range(len(points)):
        start_point = tuple(points[i])
        end_point = tuple(points[(i + 1) % len(points)])
        cv2.line(image, start_point, end_point, color, thickness)

def show(df, image_id, json_path, class_colors):
    image_data = df[df['image_id'] == image_id]

    base_path = '/'.join(json_path.split('/')[:-2]) + "/img/train/"
    image = cv2.imread(base_path + image_id)
    print(base_path + image_id)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    display_w, display_h = image.shape[1], image.shape[0]
    for _, row in image_data.iterrows():
        try:
            img_w, img_h = row['img_w'], row['img_h']
            color = class_colors.get('word', (255, 255, 255))
            scaled_coordinates = [
                int(row['x1'] * display_w / img_w), int(row['y1'] * display_h / img_h),
                int(row['x2'] * display_w / img_w), int(row['y2'] * display_h / img_h),
                int(row['x3'] * display_w / img_w), int(row['y3'] * display_h / img_h),
                int(row['x4'] * display_w / img_w), int(row['y4'] * display_h / img_h)
            ]
            draw_bbox(image, scaled_coordinates, color=color, thickness=2)
        except Exception as e:
            st.write(e)

    st.image(image, caption=image_id, use_column_width=True)