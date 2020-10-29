import numpy as np
from sklearn import metrics
import os

dir_path = os.path.dirname(os.getcwd())


def match(
        results,
        truths,
        num_class,
):
    """
    I for number of images.
    N for number of boxes in results.
    M for number of gtboxes in truths.

    :param results:
[
    [
        [
            [
              x, y, x, y, prob
            ] x number-of-bboxes
        ] x number-of-classes
    ] x number-of-images
]
    :param truths:
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        "pigment": int,
        "soft_deposit": int,
        'ann': {
            'bboxes': <np.ndarray> (n, 4 (xmin, ymin, xmax, ymax)),
            'labels': <np.ndarray> (n, ),
        }
    } x number-of-images
]
    :param classes:
    :return:
    """

    maxiou_confidence = []

    for cat in range(num_class):
    # each class
        image_iou_array = np.array([])
        for result, truth in zip(results, truths):
        # each image
            max_prob = 0
            for box in result[cat]:
            # each box
                x_min, y_min, x_max, y_max, prob = box[0:5]
                max_prob = max(prob, max_prob)

            # N x 1. 1 for max prob of bboxes for each image.
            image_iou_array = np.append(image_iou_array, max_prob)

        image_iou_array = image_iou_array.reshape((-1, 1))
        maxiou_confidence.append(image_iou_array)

    return maxiou_confidence


def plot(maxiou_confidence, truths, num_class):
    """
    :param maxiou_confidence: np.array
    :param num_groundtruthbox: int
    """

    return_result = [[], [], [], [], [], [], [], []]

    for i in range(num_class):
        cat_in_dataset = i
        gt = np.array([])
        for truth in truths:
            gt_labels = truth['ann']['labels']
            if cat_in_dataset in gt_labels:
                gt = np.append(gt, True)
            else:
                gt = np.append(gt, False)

        auc = metrics.roc_auc_score(
            y_true=gt,
            y_score=maxiou_confidence[i][:, 0],
        )
        fpr, tpr, roc_thresholds = metrics.roc_curve(
            y_true=gt,
            y_score=maxiou_confidence[i][:, 0],
        )

        precision, recall, prc_thresholds = metrics.precision_recall_curve(
            y_true=gt,
            probas_pred=maxiou_confidence[i][:, 0],
        )
        AP = metrics.average_precision_score(
            y_true=gt,
            y_score=maxiou_confidence[i][:, 0],
        )

        return_result[0].append(tpr)
        return_result[1].append(fpr)
        return_result[2].append(roc_thresholds)
        return_result[3].append(precision)
        return_result[4].append(recall)
        return_result[5].append(prc_thresholds)
        return_result[6].append(auc)
        return_result[7].append(AP)

    return return_result


def plot_imagewise_roc(
        results,
        truths,
        num_class,
):
    """
    :param results:
[
    [
        [
            [
              x, y, x, y, prob
            ] x number-of-bboxes
        ] x number-of-classes
    ] x number-of-images
]
    :param truths:
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        "pigment": int,
        "soft_deposit": int,
        'ann': {
            'bboxes': <np.ndarray> (n, 4 (xmin, ymin, xmax, ymax)),
            'labels': <np.ndarray> (n, ),
        }
    } x number-of-images
]
    :return:
    """

    assert len(results) == len(truths)
    maxiou_confidence = match(results, truths, num_class)
    return_result = plot(maxiou_confidence, truths, num_class)
    return return_result



