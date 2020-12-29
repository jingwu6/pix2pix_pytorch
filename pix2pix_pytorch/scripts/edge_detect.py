import numpy as np
import cv2

import argparse
import os


class EdgeDetect:
    def __init__(self, src_dir, dst_dir, low_thresh=100, hi_thresh=200):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.low_thresh = low_thresh
        self.hi_thresh = hi_thresh

    def _edge_detect(self, img_path):
        img = cv2.imread(img_path, 0)
        edges = cv2.Canny(img, self.low_thresh, self.hi_thresh)
        edges = ~edges  # flip 0 and 1
        return edges

    def _is_image(self, filename):
        # TODO: This is a naive implementation may need a better approach in the future
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

    def dir_edge_detect(self):
        # Create destination folder
        if not os.path.exists(self.dst_dir):
            os.mkdir(self.dst_dir)

        for path in os.listdir(self.src_dir):
            # relative paths
            src_path = os.path.join(self.src_dir, path)
            dst_path = os.path.join(self.dst_dir, path)

            # Save Edge Detected images
            if os.path.isfile(src_path) and self._is_image(src_path):
                edge_img = self._edge_detect(src_path)
                cv2.imwrite(dst_path, edge_img)


if __name__ == '__main__':
    # TODO: Argparser for src_dir and dst_dir
    ED = EdgeDetect('./images', './edge_images')
    ED.dir_edge_detect()
