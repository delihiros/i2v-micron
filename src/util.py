import numpy as np
from PIL import Image

def toarray(image):
    image_array = np.array(image.convert("RGB"), "f")
    s = image_array.shape
    return (image_array.reshape(s[0] * s[1] * s[2]), s)

def fromarray(image_array, shape):
    return Image.fromarray(np.uint8(image_array.reshape(shape)))

# delihiros = Image.open("resources/delihiros.png")
# delihiros_array, s = toarray(delihiros)
# fromarray(delihiros_array, s).save("delihiros-out.png")
