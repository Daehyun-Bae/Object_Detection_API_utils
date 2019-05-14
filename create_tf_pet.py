import os
import tensorflow as tf

from lxml import etree
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

# Path configuration
PATH_TO_OUTPUT = 'TFRecord_mtmc'
DATA_DIR = 'pet_dataset'
LABEL_DIR = 'label_map'
LABEL_NAME = 'pet_label_map.pbtxt'

# TODO = 'Fill proper value'

# os.environ['CUDA_VISIBLE_DEVICE'] = '0'

def get_class_name(filename):
    split_filename = filename.split('.')[0].split('_')
    class_name = '_'.join(split_filename[:-1])
    return class_name


def create_tf_example(data, label_map_dict, image_dir):
    image_path = os.path.join(image_dir, data['filename'])
    with tf.gfile.GFile(name=image_path, mode='rb') as fid:
        encoded_img = fid.read()

    height = int(data['size']['height'])
    width = int(data['size']['width'])

    # filename of the image
    filename = data['filename'].encode('utf8')
    encoded_image_data = encoded_img

    # b'jpeg' or b'png'
    image_format = 'jpeg'.encode('utf8')

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    # List of string class name of bounding box
    classes_text = []
    # List of class id of bounding box
    classes = []

    if 'object' in data:
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])

            # Normalized coordinates
            xmins.append(xmin/width)
            xmaxs.append(xmax/width)
            ymins.append(ymin/height)
            ymaxs.append(ymax/height)

            class_name = get_class_name(data['filename'])
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])
            print('class name: {}\nclass id: {}\nxmin:{}\nymin:{}'.format(
                class_name, label_map_dict[class_name], xmin, ymin))

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return tf_example


def main(_):
    # Make Output directory if not exist
    os.makedirs(PATH_TO_OUTPUT, exist_ok=True)

    data_dir = DATA_DIR
    label_mapt_path = os.path.join(LABEL_DIR, LABEL_NAME)
    label_map_dict = label_map_util.get_label_map_dict(label_mapt_path)

    image_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    examples_path = os.path.join(annotations_dir, 'trainval.txt')
    tf_output_path = os.path.join(PATH_TO_OUTPUT, 'pet_train.record')

    # read text file line by line
    # return list of example of identifiers (str)
    examples_list = dataset_util.read_examples_list(examples_path)

    # Define TFRecord writer
    writer = tf.python_io.TFRecordWriter(tf_output_path)

    for example in examples_list:
        xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
        if not os.path.exists(xml_path):
            print("Could not find {}, ignoring example".format(xml_path))
            continue
        with tf.gfile.GFile(name=xml_path, mode='r') as fid:
            xml_str = fid.read()    # returns contents of file as string
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        try:
            tf_example = create_tf_example(data,
                                           label_map_dict,
                                           image_dir)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            print("Invalid example: {}, ignoring".format(xml_path))
    writer.close()

if __name__ == '__main__':
    tf.app.run()
