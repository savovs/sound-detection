import os
from lxml import etree
import xml.etree.cElementTree as ET

slash = '\\' if os.name == 'nt' else '/'

def write_xml(path, image_size, boxes, int_label, folder='images'):
    """Writes an xml file like the ones used to train Darkflow: https://github.com/thtrieu/darkflow"""
    width, height = image_size

    file_name = path.split(slash)[-1].replace('.xml', '.jpg')

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = file_name + '.jpg'
    ET.SubElement(annotation, 'segmented').text = '0'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(int(width))
    ET.SubElement(size, 'height').text = str(int(height))
    ET.SubElement(size, 'depth').text = str(1)

    for coordinates in boxes:
        tag_object = ET.SubElement(annotation, 'object')
        ET.SubElement(tag_object, 'name').text = str(int_label)
        ET.SubElement(tag_object, 'pose').text = 'Unspecified'
        ET.SubElement(tag_object, 'truncated').text = '0'
        ET.SubElement(tag_object, 'difficult').text = '0'
        
        left, right, bottom, top = coordinates
        bbox = ET.SubElement(tag_object, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(left)
        ET.SubElement(bbox, 'ymin').text = str(bottom)
        ET.SubElement(bbox, 'xmax').text = str(right)
        ET.SubElement(bbox, 'ymax').text = str(top)

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = path + '.xml'
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

# if __name__ == '__main__':
#     """
#     for testing
#     """

#     folder = 'images'
#     img = [im for im in os.scandir('images') if '000001' in im.name][0]
#     objects = ['fidget_spinner']
#     tl = [(10, 10)]
#     br = [(100, 100)]
#     savedir = 'annotations'
#     write_xml(folder, img, objects, tl, br, savedir)