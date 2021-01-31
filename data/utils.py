import os
import xml.etree.ElementTree as ET

def create_train_imagenet_dataset(ilsvrc_path):
    data_path = os.path.join(ilsvrc_path, 'Data/CLS-LOC', 'train')
    annotations_path = os.path.join(ilsvrc_path, 'Annotations/CLS-LOC', 'train')
    
    information = list()
    identities = os.listdir(data_path)
    for idx, identity in enumerate(identities):
        #if (idx+1) % 10 == 0:
        #    print(idx+1, "ids preprocessing...")              
        identity_path = os.path.join(data_path, identity)
        try:
            images = os.listdir(identity_path)
        except:
            continue
        if len(images) == 0:
            continue

        for img in images:
            img_path = os.path.join(identity_path, img)
            info = dict()
            xml_id = img.split('.')[0] + '.xml'
            anno_path = os.path.join(annotations_path, identity, xml_id)
            info['path'] = img_path
            info['name'] = identity
            info['bbox'] = parse_annotation(anno_path)
            information.append(info)

    return information
    

def create_val_imagenet_dataset(ilsvrc_path):
    data_path = os.path.join(ilsvrc_path, 'Data/CLS-LOC', 'val')
    annotations_path = os.path.join(ilsvrc_path, 'Annotations/CLS-LOC', 'val')

    information = list()
    images = os.listdir(data_path)
    for idx, img in enumerate(images):
        #if (idx+1) % 1000 == 0:
        #    print(idx+1, "img preprocessing...")              
        img_path = os.path.join(data_path, img)
        info = dict()
        xml_id = img.split('.')[0] + '.xml'
        anno_path = os.path.join(annotations_path, xml_id)
        
        info['path'] = img_path
        info['bbox'], info['name'] = parse_annotation(annotation_path = anno_path,
                                                      mode='val') 
        information.append(info)
    
    return information
    
def parse_annotation(annotation_path, mode='train'):
    try:
        tree = ET.parse(annotation_path)
    except:
        return None
    root = tree.getroot()
    boxes = list()
    for i, object in enumerate(root.iter('object')):
        label = object.find('name').text.lower().strip()
        bbox = object.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        
    if mode != 'train':
        return boxes, label
    return boxes    