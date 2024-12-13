import torch
from torch.optim import Adam
import torchvision.models as models
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torch.nn.functional import cross_entropy
from torch_geometric.loader import DataLoader
from torchmetrics.classification import BinaryAccuracy
from torch_geometric.nn import GCNConv, GATConv
from tqdm import tqdm
from datasets import GraphImageDataset
from utils import read_config_file, FEATURE_VECTOR_LENGTH, create_contour_graph_with_thresholds, load_contour_data
import cv2
import numpy as np


class GraphContourLabeller(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mobilenetv3 = models.mobilenet_v3_small(pretrained=True)
        self.encoder = create_feature_extractor(mobilenetv3, ['classifier.0'])
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.conv1 = GCNConv(3 + 1024+33, 512) 
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, 128)
        self.predictor = nn.Linear(512+256+128, 18)

    def forward(self, data):
        x, edge_index, patches, document_types = data.x, data.edge_index, data.patches, data.document_types
        patches = patches.permute(0, 3, 1, 2) 

        encoded_patches = self.encoder(patches)['classifier.0']

        x = torch.cat([x, encoded_patches, document_types], dim=-1)
        out_1 = self.conv1(x, edge_index)
        out_2 = self.conv2(out_1, edge_index)
        out_3 = self.conv3(out_2, edge_index)
        out = self.predictor(torch.cat([out_1, out_2, out_3], dim=-1))

        return torch.nn.functional.softmax(out, dim=-1)

if __name__ == "__main__":
    image_path = "mixed_original_images/012020230231.jpeg"
    json_path = "mixed_dataset/mask_012020230231.json"
    image = cv2.imread(image_path)
    contours = load_contour_data(json_path)
    
    image_width = contours["imageWidth"]
    image_height = contours["imageHeight"]
    image = cv2.resize(image, (image_width, image_height))
    document_types = contours["documentTypes"]
    data = create_contour_graph_with_thresholds(contours, image, 80, 100, document_types)
    model = GraphContourLabeller()
    out = model(data)
    print(out.shape)
