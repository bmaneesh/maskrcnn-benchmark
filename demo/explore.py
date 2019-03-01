import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


if __name__ == '__main__':
    image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
    # compute predictions
    predictions, contours, _labels = coco_demo.run_on_opencv_image(image)
    labels = []
    for l in _labels:
        labels.append(coco_demo.CATEGORIES[l])
    cv2.imwrite('out.png', predictions)
