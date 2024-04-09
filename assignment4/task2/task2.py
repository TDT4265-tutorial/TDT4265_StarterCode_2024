import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x_intersect1 = gt_box[2] - prediction_box[0]
    x_intersect2 = prediction_box[2] - gt_box[0]
    y_intersect1 = gt_box[3] - prediction_box[1]
    y_intersect2 = prediction_box[3] - gt_box[1]

    # Check if they don't overlap at all
    if min(x_intersect1, x_intersect2, y_intersect1, y_intersect2) < 0:
        return 0
    # Compute intersection
    x_overlap = min(x_intersect2, x_intersect1)
    y_overlap = min(y_intersect1, y_intersect2)

    intersect = x_overlap*y_overlap
    union = (prediction_box[2] - prediction_box[0])*(prediction_box[3] - prediction_box[1]) + (gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1]) - intersect
    # Compute union
    iou = intersect/union

    #print(f'{gt_box=}\n\n{prediction_box=}\n\n{iou=}')
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    return num_tp/(num_tp+num_fp)
    raise NotImplementedError


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    return num_tp/(num_tp+num_fn)
    raise NotImplementedError


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    iou_stats = np.zeros((prediction_boxes.shape[0], gt_boxes.shape[0]))
    for i in range(prediction_boxes.shape[0]):
        for j in range(gt_boxes.shape[0]):
            iou = calculate_iou(prediction_boxes[i], gt_boxes[j])
            #print(f'Returned: {iou=}')
            iou_stats[i, j] = iou
    
    matches = []
    while True:
        if iou_stats.shape[0] == 0 or iou_stats.shape[1] == 0:
            break
        bestMach = np.argmax(iou_stats, axis=-1)
        if iou_stats.shape == (1, 1):
            bestMach = [0, 0]
        elif iou_stats.shape[0] == 1:
            bestMach = [0, bestMach]
        else:
            best_yet = 0
            i = 0
            for j in range(iou_stats.shape[0]):
                if iou_stats[j, bestMach[j]] > best_yet:
                    best_yet = iou_stats[j, bestMach[j]]
                    i = j
            #sprint(f'{iou_stats.shape=}\n{iou_stats=}\n\n\n{bestMach=}')
            bestMach = [i, bestMach[i]]
            # print(f'{iou_stats.shape=}\n{iou_stats=}\n\n\n{bestMach=}')
        #print(f'{iou_stats=}\n')
        #print(f'iou_stats[{bestMach[0]}, {bestMach[1]}]={iou_stats[bestMach[0], bestMach[1]]}\n')
        if iou_stats[bestMach[0], bestMach[1]] < iou_threshold:
            break
        matches.append(bestMach)
        iou_stats[bestMach[0], :] = 0
        iou_stats[:, bestMach[1]] = 0
    
    out_pred = np.zeros((len(matches), 4))
    out_gt = np.zeros((len(matches), 4))

    k = 0
    for (i, j) in matches:
        out_pred[k] = prediction_boxes[i]
        out_gt[k] = gt_boxes[j]
        k += 1

    return out_pred, out_gt


    # Sort all matches on IoU in descending order

    # Find all matches with the highest IoU threshold



    return np.array([]), np.array([])


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    out_pred, out_gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    tp = out_pred.shape[0]
    fp = prediction_boxes.shape[0]-tp
    fn = gt_boxes.shape[0] - tp
    # print(f'{iou_threshold=}\n')
    # print({"true_pos": tp, "false_pos": fp, "false_neg": fn})

    return {"true_pos": tp, "false_pos": fp, "false_neg": fn}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(all_prediction_boxes)):
        json = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        tp += json["true_pos"]
        fp += json["false_pos"]
        fn += json["false_neg"]
        
    if tp+fp != 0:
        precision = tp/(tp+fp)
    else: 
        precision = 1

    if tp+fn != 0:
        recall = tp/(tp+fn)
    else: 
        recall = 1
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []

    for ct in confidence_thresholds:
        prediction_boxes_subset = [np.array([all_prediction_boxes[j][i] for i in range(len(all_prediction_boxes[j])) if confidence_scores[j][i] >= ct]) for j in range(len(all_prediction_boxes))]
        #print(f'{prediction_boxes_subset=}')
        (precision, recall) = calculate_precision_recall_all_images(prediction_boxes_subset, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(1.0, 0.0, 11)
    recalls = np.flip(recalls)
    precisions = np.flip(precisions)
    print(f'{precisions=}')
    print(f'\n{recalls=}')

    print(f'{np.all(np.diff(recalls) >= 0)=}')
    '''
    new_recals = [1] + [i for i in recalls]
    new_precisions = [0] + [i for i in precisions]
    print(f'{new_precisions=}')
    print(f'\n{new_recals=}')

    current_p = new_precisions[0]
    

    # YOUR CODE HERE
    average_precision = 0
    current_index=0
    max_precision_yet = new_precisions
    max_yet = new_precisions[0]
    for i in range(len(max_precision_yet)):
        if new_precisions[i] > max_yet:
            max_yet = new_precisions[i]
        if max_precision_yet[i] < max_yet:
            max_precision_yet[i] = max_yet

    new_precisions = max_precision_yet

    for recall in recall_levels:
        while current_index + 1 < len(new_recals) and new_recals[current_index + 1] >= recall:
            current_index += 1
        
        if new_recals[current_index] == recall or current_index + 1 == len(new_recals):
            average_precision += new_precisions[current_index]
        else:
            average_precision += new_precisions[current_index]
    average_precision /= 11
    '''

    interp = np.interp(recall_levels, recalls, precisions, right=0)
    #interp = np.interp(recall_levels, recalls, precisions)
    average_precision = sum(interp)/11
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
