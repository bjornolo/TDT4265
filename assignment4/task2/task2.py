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
    x_left = max(prediction_box[0],gt_box[0])
    y_top = max(prediction_box[1], gt_box[1])
    x_right = min(prediction_box[2], gt_box[2])
    y_bottom = min(prediction_box[3], gt_box[3])
    # Compute intersection
    if x_right < x_left or y_bottom < y_top:
      # If the rectangles don't intersect, return 0
      return 0.0
    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # Compute union
    prediction_box_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    iou = intersection_area / float(prediction_box_area + gt_box_area - intersection_area)
    #END OF YOUR CODE

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
    # YOUR CODE HERE
    if (num_tp+num_fp)==0:
        return 1
    else:
        return num_tp/(num_tp+num_fp)
    #END OF YOUR CODE

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
    # YOUR CODE HERE
    if (num_tp+num_fn)==0:
        return 0
    else:
        return num_tp/(num_tp+num_fn)

    #END OF YOUR CODE

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
        iou_threshold: float threshold for a match to be made 
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # YOUR CODE HERE

    # Find all possible matches with a IoU >= iou threshold
    # Sort all matches on IoU in descending order
    # Find all matches with the highest IoU threshold

    # Initialize the match arrays
    # Sort the predicted boxes by decreasing area

    # Initialize match lists
    prediction_matches = np.zeros(len(prediction_boxes), dtype=bool)
    gt_matches = np.zeros(len(gt_boxes), dtype=bool)

    # Calculate iou between all pairs of boxes
    iou_matrix = np.zeros((len(prediction_boxes), len(gt_boxes)))
    for i, prediction_box in enumerate(prediction_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(prediction_box, gt_box)

    # Iterate through boxes in descending order of iou
    sorted_indices = np.argsort(iou_matrix.flatten())[::-1]
    for index in sorted_indices:
        i, j = np.unravel_index(index, iou_matrix.shape)
        if iou_matrix[i, j] < iou_threshold:
            break  # Stop iterating if iou falls below threshold

        # If the boxes aren't already matched, add them to the match lists
        if not prediction_matches[i] and not gt_matches[j]:
            prediction_matches[i] = True
            gt_matches[j] = True

    # Extract matched boxes
    matched_prediction_boxes = prediction_boxes[prediction_matches]
    matched_gt_boxes = gt_boxes[gt_matches]

    return matched_prediction_boxes, matched_gt_boxes
    #END OF YOUR CODE



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
    matched_pred_boxes,matched_gt_boxes=get_all_box_matches(prediction_boxes,gt_boxes,iou_threshold)
    num_tp=np.shape(matched_gt_boxes)[0]
    num_fp=np.shape(prediction_boxes)[0]-num_tp
    num_fn=np.shape(gt_boxes)[0]-num_tp
    return({"true_pos":num_tp,"false_pos":num_fp,"false_neg":num_fn})
    #END OF YOUR CODE

    raise NotImplementedError


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
    # YOUR CODE HERE
    precision=0
    recall=0
    for image in range (len(all_gt_boxes)):
      Dict=calculate_individual_image_result(all_prediction_boxes[image],all_gt_boxes[image],iou_threshold)
      precision+=calculate_precision(Dict["true_pos"],Dict["false_pos"],Dict["false_neg"])
      recall+=calculate_recall(Dict["true_pos"],Dict["false_pos"],Dict["false_neg"])
    return (precision/len(all_gt_boxes),recall/len(all_gt_boxes))
    #END OF YOUR CODE

    raise NotImplementedError


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
    # Loop trough all the different confidence levels
    for conf_tresh in confidence_thresholds:
        p_box_image = []
        # Get the prediction boxes that with more confidence than the treshold
        for i, pd_box in enumerate(all_prediction_boxes):
            p_boxes = []
            for j in range(len(pd_box)):
                if confidence_scores[i][j] >= conf_tresh:
                    p_boxes.append(all_prediction_boxes[i][j])

            p_box_image.append(np.array(p_boxes))

        # Find precision and recall for the images, with prediction boxes over confidence treshold
        precision, recall = calculate_precision_recall_all_images(
            p_box_image, all_gt_boxes, iou_threshold)

        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls)
  
    # END OF YOUR CODE

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
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = 0

    print("length of precisions: ", len(precisions))

    for recall_level in recall_levels:
        precision = 0

        for prec, rec in zip(precisions, recalls):

            if prec > precision and rec >= recall_level:
                precision = prec

        average_precision += precision
    #END OF YOUR CODE

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

def main():
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

if __name__ == "__main__":
    main()
