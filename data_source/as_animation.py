from PIL import Image
from glob import glob

images = [Image.open(i).convert("P", palette=Image.ADAPTIVE, colors=256) for i in sorted(glob("data_source/source/2020/*"))]
images = [i.resize((800,600), Image.BICUBIC) for i in images]
images[0].save("data_source/2020.gif", save_all=True, append_images=images[1:],  duration=300, loop=0)
