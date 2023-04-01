import os

from retinaface import Retinaface

retinaface = Retinaface(1)

list_dir = os.listdir("face_dataset")
image_paths = []
names = []
for name in list_dir:
    image_paths.append("face_dataset/"+name)
    names.append(name.split("_")[0])
print(list_dir)
retinaface.encode_face_dataset(image_paths,names)
