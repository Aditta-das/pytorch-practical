import torch
from collections import Counter
from IoU import intersection_over_union

def mean_avarage_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
	"""
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    # list storing all AP for respective classes
	avarage_precisions = []
    # used for numerical stability later on
	epsilon = 1e-6
	for c in range(num_classes):
		# Go through all predictions and targets,
	    # and only add the ones that belong to the
	    # current class c
		detections = [detection for detection in pred_boxes if detection[1] == c]
		ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]

	    # find the amount of bboxes for each training example
	    # Counter here finds how many ground truth bboxes we get
	    # for each training example, so let's say img 0 has 3,
	    # img 1 has 5 then we will obtain a dictionary with:
	    # amount_bboxes = {0:3, 1:5}
		amount_bboxes = Counter([gt[0] for gt in ground_truths])
	    # We then go through each key, val in this dictionary
	    # and convert to the following (w.r.t same example):
	    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
		for key, val in amount_bboxes.items():
			amount_bboxes[key] = torch.zeros(val)   

		# sort by box probabilities which is index 2
		detections.sort(key=lambda x: x[2], reverse=True)
		TP = torch.zeros(len(detections))
		FP = torch.zeros(len(detections))
		total_true_bboxes = len(ground_truths)

		# if none exists for this class then we can safely skip
		if total_true_bboxes == 0:
			continue

		for detection_idx, detection in enumerate(detections):
			# only take out ground_truths thar have the same
			# training idx as detection
			ground_truths_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
			num_gts = len(ground_truths_img)
			best_iou = 0

			for idx, gt in enumerate(ground_truths_img):
				iou = intersection_over_union(
					torch.tensor(detection[3:], 
					torch.tensor(gt[3:])),
					box_format=box_format,)
				if iou > best_iou:
					best_iou = iou
					best_gt_idx = idx

				if best_iou > iou_threshold:
					# only detect ground truth detection once
					if amount_bboxes[detection[0]][best_gt_idx] == 0:
						# true positive and add this bounding box to seen
						TP[detection_idx] = 1
						amount_bboxes[detection[0]][best_gt_idx] = 1
					else:
						FP[detection_idx] = 1
				# if iou is lower then the detection is a false positive
				else:
					FP[detection_idx] = 1
			TP_cumsum = torch.cumsum(TP, dim=0)
			FP_cumsum = torch.cumsum(FP, dim=0)
			recalls = TP_cumsum / (total_true_bboxes + epsilon)
			precisions = TP_cumsum / (TP_cumsum +FP_cumsum + epsilon)
			precisions = torch.cat((torch.tensor([1]), precisions))
			recalls = torch.cat((torch.tensor([0]), recalls))
			avarage_precisions.append(torch.trapz(precisions, recalls))

		return sum(avarage_precisions) / len(avarage_precisions)


print(mean_avarage_precision(pred_boxes=torch.tensor([
				[4, 1, 0.90, 190,380,(190+300),(380+150)],
              	[5, 7,  0.98, 300,420,(300+150),(420+210)],
              	[6, 4, 0.82,320,360,(320+200),(360+230)],
              	[7, 2, 0.87,390,50,(390+300),(50+330)],
              	[8, 5, 0.98,490,45,(490+200),(45+500)],
              	[9, 3, 0.82,480,130,(480+150),(130+400)]], dtype=torch.float32)
, true_boxes=torch.tensor([
				[4, 1, 0.90, 190,380,(190+300),(380+150)],
              	[5, 7,  0.98, 300,420,(300+150),(420+210)],
              	[6, 4, 0.82,320,360,(320+200),(360+230)],
              	[7, 2, 0.87,390,50,(390+300),(50+330)],
              	[8, 5, 0.98,490,45,(490+200),(45+500)],
              	[9, 3, 0.82,480,130,(480+150),(130+400)]])
))