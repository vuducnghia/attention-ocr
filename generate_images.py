from PIL import Image, ImageDraw, ImageFont
import random
from glob import glob
import numpy as np
import cv2

image_height = 80
image_width = 320
folder_label = 'train.txt'
def random_font():
    fontname = random.choice(list(glob('fonts/*.ttf')))
    font = ImageFont.truetype(fontname, size=random.randint(24, 32))
    return font


def random_background(height, width):
    background_image = random.choice(list(glob('background/b1.jpg')))
    original = Image.open(background_image)
    L = original.convert('L')
    original = Image.merge('RGB', (L, L, L))
    left = random.randint(0, original.size[0] - width)
    top = random.randint(0, original.size[1] - height)
    right = left + width
    bottom = top + height
    return original.crop((left, top, right, bottom))


def rand_padding():
    return random.randint(5, 30), random.randint(0, 15), random.randint(40, 50)


def generate_image(text):
    font = random_font()
    left_pad, top_pad1, top_pad2 = rand_padding()
    image = random_background(image_height, image_width)

    stroke_sat = int(np.array(image).mean())
    sat = int((stroke_sat + 127) % 255)
    mask = Image.new('L', (image_width, image_height))
    canvas = ImageDraw.Draw(mask)
    index1 = text.find(' ')

    canvas.text((left_pad, top_pad1), text[:index1], fill=sat, font=font, stroke_fill=stroke_sat, stroke_width=2)
    canvas.text((left_pad, top_pad2), text[index1:], fill=sat, font=font, stroke_fill=stroke_sat, stroke_width=2)

    image.paste(mask, (0, 0), mask)
    image = np.array(image)
    cv2.imwrite(f'images/{text}.jpg', image)

with open(folder_label, 'r') as f:
    for line in f.readlines():
        if ';' in line:
            _, txt = line.split(sep=';', maxsplit=1)
            txt = txt.strip()
            generate_image(txt)