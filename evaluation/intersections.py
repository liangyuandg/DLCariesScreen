def cal_IoU(detectedbox, groundtruthbox):
    """
    :param detectedbox: list, [leftx_det, topy_det, width_det, height_det, confidence]
    :param groundtruthbox: list, [leftx_gt, topy_gt, width_gt, height_gt, 1]
    :return iou:
    """
    leftx_det, topy_det, width_det, height_det, _ = detectedbox
    leftx_gt, topy_gt, width_gt, height_gt, _ = groundtruthbox

    centerx_det = leftx_det + width_det / 2
    centerx_gt = leftx_gt + width_gt / 2
    centery_det = topy_det + height_det / 2
    centery_gt = topy_gt + height_gt / 2

    distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
    distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

    if distancex <= 0 and distancey <= 0:
        intersection = distancex * distancey
        union = width_det * height_det + width_gt * height_gt - intersection
        iou = intersection / union
        return iou
    else:
        return 0


def cal_IoBB(detectedbox, groundtruthbox):
    """
    :param detectedbox: list, [leftx_det, topy_det, width_det, height_det, confidence]
    :param groundtruthbox: list, [leftx_gt, topy_gt, width_gt, height_gt, 1]
    :return iobb:
    """
    leftx_det, topy_det, width_det, height_det, _ = detectedbox
    leftx_gt, topy_gt, width_gt, height_gt, _ = groundtruthbox

    centerx_det = leftx_det + width_det / 2
    centerx_gt = leftx_gt + width_gt / 2
    centery_det = topy_det + height_det / 2
    centery_gt = topy_gt + height_gt / 2

    distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
    distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

    if distancex <= 0 and distancey <= 0:
        intersection = distancex * distancey
        union = width_det * height_det
        iou = intersection / union
        return iou
    else:
        return 0


def cal_IoGT(detectedbox, groundtruthbox):
    """
    :param detectedbox: list, [leftx_det, topy_det, width_det, height_det, confidence]
    :param groundtruthbox: list, [leftx_gt, topy_gt, width_gt, height_gt, 1]
    :return iobb:
    """
    leftx_det, topy_det, width_det, height_det, _ = detectedbox
    leftx_gt, topy_gt, width_gt, height_gt, _ = groundtruthbox

    centerx_det = leftx_det + width_det / 2
    centerx_gt = leftx_gt + width_gt / 2
    centery_det = topy_det + height_det / 2
    centery_gt = topy_gt + height_gt / 2

    distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
    distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

    if distancex <= 0 and distancey <= 0:
        intersection = distancex * distancey
        union = width_gt * height_gt
        iou = intersection / union
        return iou
    else:
        return 0

