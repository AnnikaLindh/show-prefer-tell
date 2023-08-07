# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


# Load all relevant bbs from the raw entity data and keep track of which belong to what entity id
def load_entity_bbs(entity_data_path, keep_entity_info=False):
    # Read the json data for this image
    with open(entity_data_path, 'rt') as infile:
        raw_data = json.load(infile)

    if keep_entity_info:
        gt_entities = raw_data['all_entities']
    else:
        # Convert the entities dict into a list since we won't be using their names
        gt_entities = list(raw_data['all_entities'].values())

    # Extract the actual bounding boxes from the bb dicts
    all_bbs = np.asarray([[bb['x_min'], bb['y_min'], bb['x_max'], bb['y_max']] for bb in raw_data['all_bbs']])

    if keep_entity_info:
        # Also extract the links from each bb to its entities
        bb_entities = [bb['entities'] for bb in raw_data['all_bbs']]
    else:
        bb_entities = None

    return all_bbs, gt_entities, bb_entities


class Flickr30kImageLoader(Dataset):
    def __init__(self, example_ids, image_dir, raw_dir, keep_entity_info=False):
        self.example_ids = example_ids
        self.image_dir = image_dir
        self.raw_dir = raw_dir
        self.keep_entity_info = keep_entity_info

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        return self.load(self.example_ids[index])

    def get_example_id(self, index):
        return self.example_ids[index]

    def load(self, example_id):
        # Check if the additional annotations exist for this example
        raw_path = os_path.join(self.raw_dir, example_id + '_raw.json')
        if not os_path.exists(raw_path):
            return None

        # all_bbs = numpy array of bounding boxes; gt_entities = list of lists of indices into all_bbs
        all_bbs, gt_entities, bb_entities = load_entity_bbs(os_path.join(self.raw_dir, example_id + '_raw.json'),
                                                            self.keep_entity_info)

        # Load image for this example
        if self.image_dir is None:
            image = None
        else:
            image = cv2.imread(os_path.join(self.image_dir, example_id + '.jpg')).astype(dtype=np.float32)

        return {"example_id": example_id, "image": image, "all_bbs": all_bbs, "gt_entities": gt_entities,
                'bb_entities': bb_entities}


class CICSinkhornPreprocessingDataset(Dataset):
    """
    Loads preprocessed data for the CIC training, but prepares it differently from CIC training. Use this dataset
        for further preprocessing where the region_features are passed through the Faster R-CNN to get the VG
        classes; these classes can then be used during training time to look up their embeddings.

    Arguments:
        example_ids (list): full list of examples in the form of image ids with the annotation number at the end
        data_dir (string): directory where the data files are located
    """

    def __init__(self, example_ids, data_dir):
        self.example_ids = example_ids
        self.data_dir = data_dir

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        return self.load(self.example_ids[index])

    def get_example_id(self, index):
        return self.example_ids[index]

    def load(self, example_id):
        data = np.load(os_path.join(self.data_dir, example_id + '.npz'))
        region_and_spatial_features = data['region_features']

        if (region_and_spatial_features.ndim > 1) and (len(region_and_spatial_features) > 1):
            # Drop all the spatial features
            region_features = region_and_spatial_features[:, :-5].astype(np.float32)
            relative_coords = region_and_spatial_features[:, -5:-1]

            center_x = (relative_coords[:, 0:1] + relative_coords[:, 2:3]) * 0.5
            center_y = (relative_coords[:, 1:2] + relative_coords[:, 3:4]) * 0.5
            length_x = relative_coords[:, 2:3] - relative_coords[:, 0:1]
            length_y = relative_coords[:, 3:4] - relative_coords[:, 1:2]

            spatial_features = np.concatenate([center_x, center_y, length_x, length_y], axis=1).astype(np.float32)
        else:
            region_features = None
            spatial_features = None

        return {'example_id': example_id, 'region_features': region_features, 'spatial_features': spatial_features}
