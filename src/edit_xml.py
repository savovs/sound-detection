import os
from xml.etree import ElementTree

import params

# This file was used to fix formatting mistakes faster
# in order to not run extract.py again since it's quite slow

xml_path = os.path.dirname('../data/train/annotations/')

labels = []

for root, sub_dirs, files in os.walk(params.AUDIO_PATH):
    for file in files:
        if file.endswith(params.FILE_EXTENSION):
            folder_name = root.split('\\')[-1]

            if (folder_name != 'data' and folder_name not in labels):
                labels.append(folder_name)

labels = sorted(labels)

for root, sub_dirs, files in os.walk(xml_path):
    for file in files:
        if file.endswith('.xml'):
            file_path = os.path.join(root, file)
            dom = ElementTree.parse(file_path)
            
            names = dom.findall('object/name')
            for name in names:
                if name.text.isdigit():
                    name.text = labels[int(name.text)]

            file_names = dom.findall('filename')
            file_names[0].text = file_names[0].text + '.jpg'
            dom.write(file_path)

            
