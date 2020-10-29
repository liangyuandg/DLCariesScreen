import mmcv
from PIL import Image
import numpy as np
import json

# opencv2 coordinate
# 0/0---column (y)--->
#  |
#  |
# row (x)
#  |
#  |
#  v

# PIL coordinate
# 0/0---column/width (x)--->
#  |
#  |
# row/height (y)
#  |
#  |
#  v

# annotation.pickle coordinate
# 0/0---column/width (x)--->
#  |
#  |
# row/height (y)
#  |
#  |
#  v

# annotation.pickle label
# 0 -> negative
# >= 1 -> positives

JSON_FILE = '/PATH/TO/INPUT/JSON/FILE'
PICKLE_FILE = '/PATH/TO/OUTPUT/PICKLE/FILE'
IMG_DIR = '/PATH/TO/IMAGE/REPO/'


def main():

    setcheck = set()
    annotations = []
    with open(JSON_FILE, 'rt', encoding='utf-8') as json_file:
        items = json.load(json_file)
    print('generating annotation files for {} images'.format(len(items)))

    for item in items:
        if item['DataRow ID'] in setcheck:
            print('{} already in set'.format(item['DataRow ID']))
            continue

        filename = '{}.jpg'.format(item['DataRow ID'])
        image_file = IMG_DIR + filename
        image = Image.open(image_file)

        width, height = image.size

        bboxes = []
        labels = []
        item_labels = item['Label']
        # labels
        if item_labels is None or item_labels == 'Skip':
            continue
        else:
            if 'caries' in item_labels.keys():
                for each_finding in item_labels['caries']:
                    points = each_finding['geometry']
                    xmin = int(min([point['x'] for point in points]))
                    ymin = int(min([point['y'] for point in points]))
                    xmax = int(max([point['x'] for point in points]))
                    ymax = int(max([point['y'] for point in points]))
                    box = [xmin, ymin, xmax, ymax]
                    bboxes.append(box)
                    labels.append(1)

            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
            else:
                bboxes = np.array(bboxes, ndmin=2)
                labels = np.array(labels)

        annotation = {
            'filename': filename,
            'width': width,
            'height': height,
            'ann': {
                'bboxes': bboxes.astype(np.float32),
                'labels': labels.astype(np.int32),
            }
        }

        annotations.append(annotation)
        print('finish for file {}'.format(filename))

        if item['DataRow ID'] in setcheck:
            raise RuntimeError('{} already in the set'.format(item['DataRow ID']))
        else:
            setcheck.add(item['DataRow ID'])

    mmcv.dump(annotations, PICKLE_FILE)


if __name__ == '__main__':
    main()
