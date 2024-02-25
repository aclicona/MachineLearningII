from PIL import Image
from logistic_regression import get_digits_from_csv
import numpy as np


def to_image(numpy_img):
    img = Image.fromarray(numpy_img.astype(np.uint8))
    return img


def save_image(img, name, path='mnist/image_samples'):
    img.save(f"{path}/{name}.jpg")


_, x_test, _, y = get_digits_from_csv()

for pos, img_array in enumerate(x_test[:50]):
    image = to_image(img_array.reshape((28, -1)))
    save_image(image, f'example_{pos}')
