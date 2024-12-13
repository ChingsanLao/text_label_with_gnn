import os
from glob import glob
import torch
from torch_geometric.data import Dataset, Data
from utils import load_contour_data, create_contour_graph_with_thresholds
import cv2
import time


class GraphImageDataset(Dataset):
    def __init__(self, image_dir: str, json_dir: str, vertical_threshold: int = 80, horizontal_threshold: int = 100):
        super(GraphImageDataset, self).__init__()
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpeg")))
        self.json_paths = sorted(glob(os.path.join(json_dir, "*.json")))
        self.vertical_threshold = vertical_threshold
        self.horizontal_threshold = horizontal_threshold
        assert len(self.image_paths) == len(self.json_paths), (
            "Mismatch between the number of images and JSON files."
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        json_path = self.json_paths[idx]
        contours = load_contour_data(json_path)
        image = cv2.imread(image_path)
        image_width = contours["imageWidth"]
        image_height = contours["imageHeight"]
        document_types = contours["documentTypes"]
        image = cv2.resize(image, (image_width, image_height))
        graph = create_contour_graph_with_thresholds(
            contours, image, self.vertical_threshold, self.horizontal_threshold, document_types
        )
        return graph

if __name__ == "__main__":
    image_dir = "mixed_original_images"
    json_dir = "mixed_dataset"
    dataset = GraphImageDataset(image_dir, json_dir)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        print(f"Sample {idx}:")
        print(f"  - Nodes: {sample.x.shape}")
        print(f"  - Edges: {sample.edge_index.shape}")
        print(f"  - Labels: {sample.y}")
        print(f"  - Image: {sample.patches}")