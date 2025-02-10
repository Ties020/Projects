# Function read_inkml_file(root) from: https://github.com/google-research/google-research/blob/master/mathwriting/mathwriting_code_examples.ipynb

from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET


def read_inkml_file(root):
    strokes = []
    for element in root:
        tag_name = element.tag.removeprefix('{http://www.w3.org/2003/InkML}')
        if tag_name == 'trace':
            points = element.text.split(',')
            stroke_x, stroke_y= [], []
            for point in points:
                x, y, t = point.split(' ')
                stroke_x.append(float(x))
                stroke_y.append(float(y))
            strokes.append(np.array((stroke_x, stroke_y)))
    return strokes


def inkml_to_image(inkml, img_width=128, img_height=128, stroke_width=9):
    # Create image to draw on
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Get minimum and maximum x and y to normalize data
    trace_data = read_inkml_file(inkml)
    x = []
    y = []
    for elem in trace_data:
        x.extend(elem[0])
        y.extend(elem[1])
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    scale_x = img_width / (max_x - min_x) if max_x - min_x != 0 else 1
    scale_y = img_height / (max_y - min_y) if max_y - min_y != 0 else 1
    # Loop through traces and draw corresponding lines
    for trace in trace_data:
        for i in range(1, len(trace[0])):
            x1 = trace[0][i-1]
            y1 = trace[1][i-1]
            x2 = trace[0][i]
            y2 = trace[1][i]
            x1s = int((x1 - min_x) * scale_x)
            y1s = int((y1 - min_y) * scale_y)
            x2s = int((x2 - min_x) * scale_x)
            y2s = int((y2 - min_y) * scale_y)
            draw.line((x1s, y1s, x2s, y2s), fill=(0, 0, 0), width=stroke_width)
    return img