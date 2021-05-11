import streamlit as st
st.set_page_config(layout="wide", page_title="Socialis", page_icon="")
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.title("Object Detection on Cloud")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


local_css("style.css")


@st.cache(show_spinner=False)
def read_img(img):
    input_image = cv2.imread(img, cv2.IMREAD_COLOR)
    input_image = input_image[:, :, [2, 1, 0]]
    return input_image


def yolo_v3(ip_image, conf=0.5, overlap_threshold=0.3):
    net = cv2.dnn.readNetFromDarknet("config.cfg", "model.weights")
    output_layer_names = net.getLayerNames()
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(ip_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    boxes, confidences, class_ids = [], [], []
    h, w = ip_image.shape[:2]

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf:
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype("int")
                x, y = int(center_x - (width / 2)), int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    f = open("coco.names", "r")
    f = f.readlines()
    f = [line.rstrip('\n') for line in list(f)]
    try:
        st.subheader("Detected Objects: " + ', '.join(list(set([f[obj] for obj in class_ids]))))
    except IndexError:
        st.write("Nothing Detected D:")

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, overlap_threshold)
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            xmin.append(x)
            ymin.append(y)
            xmax.append(x + w)
            ymax.append(y + h)
    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    label_colors = [0, 255, 0]
    image_with_boxes = ip_image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] += label_colors
        image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] /= 2
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)


img_type = st.sidebar.selectbox("Select image type", ['Browse', 'Cars', 'Person', 'Animals'], 1)
st.sidebar.write("Made by Lavin, Parth and Dharmik  \n\n Group 42 BE-C2")
image_url = ""
if img_type == 'Browse':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        new_img = np.array(our_image.convert('RGB'))
        image = cv2.cvtColor(new_img, 1)
        yolo_v3(image, confidence_threshold)
    else:
        st.markdown("Please upload an image")

elif img_type == 'Person':
    image_url = "images/person.jpg"
    image = read_img(image_url)
    yolo_v3(image, confidence_threshold)
elif img_type == 'Cars':
    image_url = "images/cars.jpg"
    image = read_img(image_url)
    yolo_v3(image, confidence_threshold)
elif img_type == 'Animals':
    image_url = "images/animals.jpg"
    image = read_img(image_url)
    yolo_v3(image, confidence_threshold)
