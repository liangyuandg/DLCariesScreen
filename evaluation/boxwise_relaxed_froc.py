import numpy as np
import os
from evaluation.intersections import cal_IoBB, cal_IoGT, cal_IoU

dir_path = os.path.dirname(os.getcwd())


def match(
        results,
        truths,
        IOX_method,
        IOX_true_threshold,
        num_class,
        IoRelaxed,
        reuse_box,
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

    maxiou_match = [[] for i in range(num_class)]
    maxiou_confidence = [[] for i in range(num_class)]

    for cat in range(num_class):
    # each class
        cat_wise_match = [] 
        cat_wise_confidence = []
        for result, truth in zip(results, truths):
        # each image
            cat_in_result = cat
            label_in_dataset = cat
            num_gt = np.sum(truth['ann']['labels'] == label_in_dataset)

            image_wise_confidence = np.zeros(len(result[cat_in_result]))
            if num_gt == 0:
                image_wise_match = np.zeros([1, len(result[cat_in_result])])
            else:
                image_wise_match = np.zeros([num_gt, len(result[cat_in_result])])

            box_count = 0
            for box in result[cat_in_result]:
            # each box
                x_min, y_min, x_max, y_max, prob = box[0:5]
                # N. 1 for prob.
                image_wise_confidence[box_count] = prob
                gt_count = 0
                for truth_box, truth_label in zip(truth['ann']['bboxes'], truth['ann']['labels']):
                    if truth_label != label_in_dataset:
                        continue

                    if IoRelaxed is False:
                        iou = cal_IoU(
                            detectedbox=[x_min, y_min, x_max-x_min+1, y_max-y_min+1, prob],
                            groundtruthbox=[truth_box[0], truth_box[1], truth_box[2]-truth_box[0]+1, truth_box[3]-truth_box[1]+1, 1],
                        )
                    else:

                        if IOX_method is 'IOU':
                            iou = cal_IoU(
                                detectedbox=[x_min, y_min, x_max-x_min+1, y_max-y_min+1, prob],
                                groundtruthbox=[truth_box[0], truth_box[1], truth_box[2]-truth_box[0]+1, truth_box[3]-truth_box[1]+1, 1],
                            )
                        elif IOX_method is 'IOG':
                            iou = cal_IoGT(
                                detectedbox=[x_min, y_min, x_max-x_min+1, y_max-y_min+1, prob],
                                groundtruthbox=[truth_box[0], truth_box[1], truth_box[2]-truth_box[0]+1, truth_box[3]-truth_box[1]+1, 1],
                            )
                        elif IOX_method is 'IOF':
                            iou = cal_IoBB(
                                detectedbox=[x_min, y_min, x_max-x_min+1, y_max-y_min+1, prob],
                                groundtruthbox=[truth_box[0], truth_box[1], truth_box[2]-truth_box[0]+1, truth_box[3]-truth_box[1]+1, 1],
                            )
                        elif IOX_method is 'ALL':
                            iou = max(
                                cal_IoBB(
                                    detectedbox=[x_min, y_min, x_max-x_min+1, y_max-y_min+1, prob],
                                    groundtruthbox=[truth_box[0], truth_box[1], truth_box[2]-truth_box[0]+1, truth_box[3]-truth_box[1]+1, 1],
                                ),
                                cal_IoGT(
                                    detectedbox=[x_min, y_min, x_max-x_min+1, y_max-y_min+1, prob],
                                    groundtruthbox=[truth_box[0], truth_box[1], truth_box[2]-truth_box[0]+1, truth_box[3]-truth_box[1]+1, 1],
                                ),
                                cal_IoU(
                                    detectedbox=[x_min, y_min, x_max-x_min+1, y_max-y_min+1, prob],
                                    groundtruthbox=[truth_box[0], truth_box[1], truth_box[2]-truth_box[0]+1, truth_box[3]-truth_box[1]+1, 1],
                                )
                            )
                        else:
                            raise RuntimeError('IOX_method {} not accepted. '.format(IOX_method))

                    if iou >= IOX_true_threshold:
                        image_wise_match[gt_count][box_count] = prob
                    else:
                        pass
                    gt_count = gt_count + 1

                box_count = box_count + 1

            for row in range(num_gt):
                # no prediction
                if len(image_wise_match[row, :]) == 0:
                    max_index = 0
                    temp_value = 0
                else:
                    max_index = np.argmax(image_wise_match[row, :])
                    temp_value = image_wise_match[row, max_index]

                if temp_value == 0 or temp_value == -1:
                    # no bbox match a gt.
                    psudo_bbox = np.zeros([num_gt, 1])
                    psudo_bbox[row, 0] = 1
                    image_wise_match = np.append(image_wise_match, psudo_bbox, axis=1)
                    # psudo bbox of prob = 0
                    image_wise_confidence = np.append(image_wise_confidence, 0.0)
                else:
                    # a bbox marked as match for a gt.
                    # bboxes has overlap marked as ignored (-1)
                    bboxes_of_overlap = (image_wise_match[row, :] > 0)
                    image_wise_match[row, bboxes_of_overlap] = -1
                    if reuse_box is True:
                        pass
                    else:
                        image_wise_match[:, max_index] = 0
                    image_wise_match[row, max_index] = 1

            # 1 x N. 1 for 1 if matched, 0 is not matched, -1 for ignored.
            # if there's 1, then 1. if there is -1, then -1. else 0.
            new_image_wise_match = []
            for column_index in range(image_wise_match.shape[1]):
                if 1 in image_wise_match[:, column_index]:
                    new_image_wise_match.append(1)
                elif -1 in image_wise_match[:, column_index]:
                    new_image_wise_match.append(-1)
                else:
                    new_image_wise_match.append(0)
            new_image_wise_match = np.array(new_image_wise_match)
            # number_of_images x N.
            cat_wise_match.append(new_image_wise_match)
            # number_of_images x N.
            cat_wise_confidence.append(image_wise_confidence)
        # K x number_of_images x N.
        maxiou_match[cat] = cat_wise_match
        # K x number_of_images x N.
        maxiou_confidence[cat] = cat_wise_confidence

    return maxiou_match, maxiou_confidence


def plot(maxiou_match, maxiou_confidence, num_class, num_images):
    """
    :param maxiou_match: K x number_of_images x N.
    :param maxiou_confidence: K x number_of_images x N.
    """

    return_results = [[], [], []]

    all_tp_list = [[] for i in range(num_class)]
    all_fp_list = [[] for i in range(num_class)]
    all_fn_list = [[] for i in range(num_class)]

    for i in range(num_class):

        gt = np.concatenate(maxiou_match[i])
        confidence = np.concatenate(maxiou_confidence[i])

        non_ignore_list = (gt != -1)
        gt = gt[non_ignore_list]
        confidence = confidence[non_ignore_list]

        tp_list = []
        fp_list = []
        fn_list = []
        threshold_list = (np.linspace(start=0.0, stop=1.0, num=1001, endpoint=True)).tolist()

        for threshold in threshold_list:
            tp = np.sum(np.logical_and(confidence >= threshold, gt == 1))
            fp = np.sum(np.logical_and(confidence >= threshold, gt == 0))
            fn = np.sum(np.logical_and(confidence < threshold, gt == 1))

            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            all_tp_list[i].append(tp)
            all_fp_list[i].append(fp)
            all_fn_list[i].append(fn)

        tpr = [tp / (fn + tp) for tp, fn in zip(tp_list, fn_list)]
        avg_fp = np.array(fp_list) / num_images

        # plot
        return_results[0].append(tpr)
        return_results[1].append(avg_fp)
        return_results[2].append(threshold_list)

    return return_results


def plot_boxwise_relaxed_froc(
        results,
        truths,
        IOX_method,
        IOX_true_threshold,
        num_class,
        IoRelaxed,
        reuse_box=False,
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
    assert IOX_method in ['IOU', 'IOF', 'IOG', 'ALL']
    maxiou_match, maxiou_confidence = match(results, truths, IOX_method, IOX_true_threshold, num_class, IoRelaxed, reuse_box)
    return_results = plot(maxiou_match, maxiou_confidence, num_class, len(truths))

    return return_results



