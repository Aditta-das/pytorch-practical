import torch
from IoU import intersection_over_union

def nms(bboxes, iou_threshold, threshold, box_format="corners"):
	"""
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
	assert type(bboxes) == list
	bboxes = [box for box in bboxes if box[1] > threshold]
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
	bboxes_after_nms = []

	while bboxes:
		chosen_box = bboxes.pop(index=0)
		bboxes = [box for box in bboxes 
							if box[0] != chosen_box[0] or intersection_over_union
							(torch.tensor(chosen_box[2:]), 
								torch.tensor(chosen_box[2:]),
			 					box_format="midpoint") < iou_threshold]
		bboxes_after_nms.append(chosen_box)

	return bboxes_after_nms


	
def test():
	bboxes = torch.tensor([
				[1, 0.90, 190,380,(190+300),(380+150)],
              	[7, 0.98, 300,420,(300+150),(420+210)],
              	[4, 0.82,320,360,(320+200),(360+230)],
              	[2, 0.87,390,50,(390+300),(50+330)],
              	[5, 0.98,490,45,(490+200),(45+500)],
              	[3, 0.82,480,130,(480+150),(130+400)]], dtype=torch.float32)
	threshold = torch.tensor([[0.9]], dtype=torch.float32)
	bboxes = [box for box in bboxes if box[1] > threshold]
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
	bboxes_after_nms = []
	iou_threshold = 0.5

	while bboxes:
		chosen_box = bboxes.pop(0)
		bboxes = [
			box for box in bboxes
			if box[0] != chosen_box[0] or
			intersection_over_union(
				torch.tensor(chosen_box[2:]),
				torch.tensor(box[2:]),
			) < iou_threshold
		]
		bboxes_after_nms.append(chosen_box)
	print(bboxes_after_nms)


test()