import i2v_micron
from PIL import Image

i2v_model = i2v_micron.make_i2v_micron(
        "illust2vec_tag_ver200.caffemodel", "tag_list.json")

img = Image.open("resources/miku.png")
print(i2v_model.estimate_specific_tags([img], ["hatsune miku"]))
