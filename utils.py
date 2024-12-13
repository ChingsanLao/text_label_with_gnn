import json
import cv2
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import scipy.spatial as spatial
import torch_geometric.transforms as T
import os


# Constants
FEATURE_VECTOR_LENGTH = 3

def read_config_file(config_file):
    """
    Read a configuration file in JSON format.
    
    :param config_file: Path to the configuration file
    :return: Dictionary with configuration parameters
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_contour_data(json_path):
    """
    Load contour data from a JSON file.
    
    Expected JSON structure:
    {
        "shapes": [
            {
                "points": [[x1, y1], [x2, y2]],
                "label": "optional_label"
            },
            ...
        ]
    }
    """
    with open(json_path, 'r') as f:
        contours = json.load(f)   
    return contours

def calculate_contour_center(contour):
    """
    Calculate the center point of a contour.
    
    :param contour: List of points defining the contour
    :return: Center point (x, y)
    """
    center_x = int((contour[0][0] + contour[1][0]) / 2)
    center_y = int((contour[0][1] + contour[1][1]) / 2)
    return center_x, center_y

def find_boundary_points(contour):
    """
    Find boundary points (top, bottom, left, right) of a contour.
    
    :param contour: List of points defining the contour
    :return: Dictionary of boundary points
    """
    top = [np.inf, np.inf]
    bottom = [-np.inf, -np.inf]
    left = [np.inf, np.inf]
    right = [-np.inf, -np.inf]
    
    for point in contour:
        x, y = point
        if x < left[0]:
            left = point
        if x > right[0]:
            right = point
        if y < top[1]:
            top = point
        if y > bottom[1]:
            bottom = point
    
    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}

def is_point_between_vertically(point, ref_point1, ref_point2):
    """
    Check if a point is vertically between two reference points.
    """
    min_y = min(ref_point1[1], ref_point2[1])
    max_y = max(ref_point1[1], ref_point2[1])
    return min_y <= point[1] <= max_y

def is_point_between_horizontally(point, ref_point1, ref_point2):
    """
    Check if a point is horizontally between two reference points.
    """
    min_x = min(ref_point1[0], ref_point2[0])
    max_x = max(ref_point1[0], ref_point2[0])
    return min_x <= point[0] <= max_x

def create_contour_graph_with_thresholds(contours, image, vertical_threshold, horizontal_threshold, document_types):
    """
    Create a graph from contours with thresholds for top/bottom and left/right connections.

    :param contours: Contour data dictionary
    :param image_path: Path to the image for patch extraction
    :param vertical_threshold: Threshold for vertical connections
    :param horizontal_threshold: Threshold for horizontal connections
    :return: PyTorch Geometric Data object with edge information and labels
    """
    shapes = contours["shapes"]
    img_height, img_width, _ = image.shape
    img_area = img_width * img_height

    contour_features = []  # To store [x, y, area] for each patch
    labels = []
    boundary_points = []
    patches = []
    for shape in shapes:
        contour = shape["points"]
        labels.append(shape.get("label", "unknown"))
        # if document_types == "PASSPORT_KH_0":
        #     print(contour)

        # Calculate the bounding box and extract the patch
        x_min, y_min = map(int, contour[0])
        x_max, y_max = map(int, contour[1])

        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        # Extract patch
        patch = image[y_min:y_max, x_min:x_max]

        patch = cv2.resize(patch, (224, 224))
        patches.append(patch)


        # Normalize x, y, area
        center_x, center_y = calculate_contour_center(contour)
        norm_x = center_x / img_width
        norm_y = center_y / img_height
        norm_area = area / img_area

        contour_features.append([norm_x, norm_y, norm_area])
        boundary_points.append(find_boundary_points(contour))

    # Build the graph edges using KDTree and spatial relationships
    tree = spatial.KDTree([[c[0] * img_width, c[1] * img_height] for c in contour_features])  # Scale back for spatial comparisons
    edge_indices = []
    edge_directions = {}

    for i, (center, bounds) in enumerate(zip(contour_features, boundary_points)):
        distances, indices = tree.query([center[0] * img_width, center[1] * img_height], k=len(shapes))

        edge_directions[i] = {
            'top': None,
            'bottom': None,
            'left': None,
            'right': None
        }

        for j, idx in zip(distances, indices):
            if i == idx or j == 0:
                continue

            other_bounds = boundary_points[idx]
            other_center = contour_features[idx]

            if (bounds['right'][0] < other_center[0] * img_width and
                abs(bounds['right'][0] - other_center[0] * img_width) <= horizontal_threshold and
                is_point_between_vertically([other_center[0] * img_width, other_center[1] * img_height],
                                            bounds['top'], bounds['bottom']) and
                edge_directions[i]['right'] is None):
                edge_indices.append([i, idx])
                edge_directions[i]['right'] = idx

            elif (bounds['left'][0] > other_center[0] * img_width and
                  abs(bounds['left'][0] - other_center[0] * img_width) <= horizontal_threshold and
                  is_point_between_vertically([other_center[0] * img_width, other_center[1] * img_height],
                                              bounds['top'], bounds['bottom']) and
                  edge_directions[i]['left'] is None):
                edge_indices.append([i, idx])
                edge_directions[i]['left'] = idx

            elif (bounds['top'][1] > other_center[1] * img_height and
                  abs(bounds['top'][1] - other_center[1] * img_height) <= vertical_threshold and
                  is_point_between_horizontally([other_center[0] * img_width, other_center[1] * img_height],
                                                bounds['left'], bounds['right']) and
                  edge_directions[i]['top'] is None):
                edge_indices.append([i, idx])
                edge_directions[i]['top'] = idx

            elif (bounds['bottom'][1] < other_center[1] * img_height and
                  abs(bounds['bottom'][1] - other_center[1] * img_height) <= vertical_threshold and
                  is_point_between_horizontally([other_center[0] * img_width, other_center[1] * img_height],
                                                bounds['left'], bounds['right']) and
                  edge_directions[i]['bottom'] is None):
                edge_indices.append([i, idx])
                edge_directions[i]['bottom'] = idx

    contour_labels = [
        "address",
        "characteristic",
        "contour",
        "country_code",
        "date_of_birth",
        "expiry_date",
        "first_name_en",
        "first_name_kh",
        "gender",
        "height",
        "id_number",
        "issue_date",
        "label",
        "last_name_en",
        "last_name_kh",
        "nationality",
        "passport_type",
        "place_of_birth"
    ]

    # [0010000], [001000]
    one_hot_labels = []
    for label in labels:
        one_hot = [1 if label == contour_label else 0 for contour_label in contour_labels]
        one_hot_labels.append(one_hot)
    labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float)

    # Create node features
    x = torch.tensor(contour_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    patches = [torch.from_numpy(patch).float() for patch in patches]
    patches = torch.stack(patches) 

    document_types_label = [
        "COVID_19_VACCINATION_CARD_0",
        "COVID_19_VACCINATION_CARD_0_BACK",
        "COVID_19_VACCINATION_CARD_1",
        "COVID_19_VACCINATION_CARD_1_BACK",
        "DRIVER_LICENSE_0",
        "DRIVER_LICENSE_0_BACK",
        "DRIVER_LICENSE_1",
        "DRIVER_LICENSE_1_BACK",
        "LICENSE_PLATE_0_0",
        "LICENSE_PLATE_0_1",
        "LICENSE_PLATE_1_0",
        "LICENSE_PLATE_2_0",
        "LICENSE_PLATE_3_0",
        "LICENSE_PLATE_3_1",
        "NATIONAL_ID_0",
        "NATIONAL_ID_0_BACK",
        "NATIONAL_ID_1",
        "NATIONAL_ID_1_BACK",
        "NATIONAL_ID_2",
        "NATIONAL_ID_2_BACK",
        "PASSPORT_0",
        "PASSPORT_CA_0",
        "PASSPORT_KH_0",
        "PASSPORT_KH_0_TOP",
        "PASSPORT_SG_0",
        "PASSPORT_INT",
        "SUPERFIT_0",
        "SUPERFIT_0_BACK",
        "VEHICLE_REGISTRATION_0",
        "VEHICLE_REGISTRATION_0_BACK",
        "VEHICLE_REGISTRATION_1",
        "VEHICLE_REGISTRATION_1_BACK",
        "VEHICLE_REGISTRATION_2",
    ]

    # hot encode document types as features:
    # One-hot encode document types as features
    document_types_tensor = torch.zeros(len(document_types_label), dtype=torch.float)
    for i, doc_type_label in enumerate(document_types_label):
        if doc_type_label in document_types:  # Assuming document_types is a list of document type strings
            document_types_tensor[i] = 1

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=labels_tensor,
        patches=patches,
        document_types=document_types_tensor.unsqueeze(0).repeat(x.size(0), 1)
    )
    graph = T.ToUndirected()(graph)
    return graph

def visualize_and_save_graph(image, contours, patches, edge_index, labels_tensor, output_dir):
    """
    Visualize the graph connections on the image and save patches with indices.
    
    :param image: Original image as a NumPy array
    :param contours: Contour data
    :param patches: List of patches (224x224 images)
    :param edge_index: Edge indices representing graph connections
    :param labels_tensor: Tensor containing the labels for each node (y values)
    :param output_dir: Directory to save the visualization and patches
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the original image for annotations
    annotated_image = image.copy()
    
    # Draw nodes and their corresponding labels
    for idx, contour in enumerate(contours["shapes"]):
        center_x, center_y = calculate_contour_center(contour["points"])
        
        # Draw node center
        cv2.circle(annotated_image, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Get the label for the node from the labels_tensor
        label = labels_tensor[idx].argmax().item()
        
        # Add label to the node
        cv2.putText(annotated_image, str(label), (center_x + 10, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save the patch with its label
        patch_path = os.path.join(output_dir, f"patch_{idx}.png")
        patch_with_label = patches[idx].numpy().astype(np.uint8)
        cv2.putText(patch_with_label, str(label), (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imwrite(patch_path, patch_with_label)

    # Draw edges
    for edge in edge_index.t().tolist():  # edge_index contains the pair of indices
        start_idx, end_idx = edge
        start_contour = contours["shapes"][start_idx]["points"]
        end_contour = contours["shapes"][end_idx]["points"]
        start_center_x, start_center_y = calculate_contour_center(start_contour)
        end_center_x, end_center_y = calculate_contour_center(end_contour)

        # Draw an edge between the nodes
        cv2.line(annotated_image, (start_center_x, start_center_y), (end_center_x, end_center_y), (255, 0, 0), 2)

    # Save the annotated image
    annotated_image_path = os.path.join(output_dir, "annotated_image.png")
    cv2.imwrite(annotated_image_path, annotated_image)


# test example
if __name__ == "__main__":
    config_path = "config.json"
    image_path = "mixed_original_images/passport.jpeg"
    json_path = "mixed_dataset/mask_passport.json"
    output_dir = "output"

    config = read_config_file(config_path)
    image = cv2.imread(image_path)
    contours = load_contour_data(json_path)

    vertical_threshold = 80
    horizontal_threshold = 100
    document_types = contours["documentTypes"]

    graph_data = create_contour_graph_with_thresholds(contours, image, vertical_threshold, horizontal_threshold, document_types)
    patches = graph_data.patches
    edge_index = graph_data.edge_index
    labels_tensor = graph_data.y

    visualize_and_save_graph(image, contours, patches, edge_index, labels_tensor, output_dir)
