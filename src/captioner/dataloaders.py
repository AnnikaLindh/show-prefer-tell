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


class PreprocessedDataset(Dataset):
    """
    Loads all the relevant data from disk and prepares it for collating.
    Only the list of all example ids is permanently stored in memory.

    Arguments:
        example_ids (list): full list of examples in the form of image ids with the annotation number at the end
        data_dir (string): directory where the data files are located
        num_regions (int): test results on ground-truth examples under specific conditions
        image_dir (string): directory where the raw pixel image files are located
    """

    def __init__(self, example_ids, data_dir, num_regions, image_dir=None):
        self.example_ids = example_ids
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.num_regions = num_regions

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        return self.load(self.example_ids[index])

    def get_example_id(self, index):
        return self.example_ids[index]

    def load(self, example_id):
        loaded_data = dict()
        loaded_data['example_id'] = example_id

        if '_' in example_id:
            data = np.load(os_path.join(self.data_dir, example_id + '.npz'))
            loaded_data['region_features'] = data['region_features']
            if self.num_regions > 0:
                loaded_data['region_features'] = loaded_data['region_features'][:self.num_regions]
            loaded_data['full_image_features'] = data['full_image_features']
        else:
            loaded_data['full_image_features'] = None
            loaded_data['region_features'] = list()
            # This is an image_id, so load the (unique) regions from ALL its examples
            unique_boxes = list()
            for i_annotation in range(5):
                try:
                    data = np.load(os_path.join(self.data_dir, example_id + '_' + str(i_annotation) + '.npz'))
                except IOError:
                    continue

                loaded_data['full_image_features'] = data['full_image_features']  # This should be the same for all

                # Only add entity regions that have not already been added
                current_boxes = data['region_features'][:, -5:-1]
                new_boxes = [i_box for i_box in range(len(current_boxes))
                             if list(current_boxes[i_box]) not in unique_boxes]
                unique_boxes.extend([list(current_boxes[i_box]) for i_box in new_boxes])
                loaded_data['region_features'].extend(data['region_features'][new_boxes, :])

        if self.image_dir is not None:
            image_id = example_id.split('_')[0]
            img = cv2.imread(os_path.join(self.image_dir, image_id + '.jpg'))
            height, width, _ = img.shape
            loaded_data['image_height'] = height
            loaded_data['image_width'] = width

        return loaded_data


class CollatePreprocessedData:
    """
    Takes a batch of data example dicts and return a dict where each entry is batched.
    The collated batch contains the following entries:
        example_ids          - list of example ids in this batch [batch_size]
        full_image_features  - torch matrix of full image features [batch_size, 2048]
        region_features      - torch matrix of region features (incl spatial features) [1 + num_all_regions, 2048 + 5]
        region_start_indices - tuple containing the indices of the first region features for each example [batch_size]
        region_end_indices   - tuple containing the indices of 1 beyond the last features for each example [batch_size]
    """
    def __init__(self, region_feature_size):
        self.region_feature_size = region_feature_size

    def __call__(self, batch):
        batch_size = len(batch)

        collated_batch = dict()

        # -- EXAMPLE_ID
        # Turn into a single list
        collated_batch['example_ids'] = [batch[i]['example_id'] for i in range(batch_size)]

        # -- FULL_IMAGE_FEATURES
        # Gather all and convert into a single torch matrix
        collated_batch['full_image_features'] = torch.tensor(
            [batch[i]['full_image_features'] for i in range(batch_size)], dtype=torch.float32).detach()

        # -- REGION_FEATURES -- REGION_START_INDICES -- REGION_END_INDICES
        # Add an empty region at the start of the region features list
        collated_batch['region_features'] = [np.zeros([self.region_feature_size])]
        # Add the region features from all examples to a single list and keep track of their start and end indices
        collated_batch['region_start_indices'] = list()
        collated_batch['region_end_indices'] = list()
        for i_batch in range(batch_size):
            feats = batch[i_batch]['region_features']
            if len(feats) > 0:
                collated_batch['region_start_indices'].append(len(collated_batch['region_features']))
                collated_batch['region_features'].extend(feats)
                collated_batch['region_end_indices'].append(len(collated_batch['region_features']))
            else:
                # If this example has no regions, start at index zero where the empty region is
                collated_batch['region_start_indices'].append(0)
                collated_batch['region_end_indices'].append(1)

        # Stack the region features from all examples and convert into a torch tensor
        collated_batch['region_features'] = torch.tensor(np.stack(collated_batch['region_features'], axis=0),
                                                         dtype=torch.float32).detach()
        # Convert the index lists into tuples
        collated_batch['region_start_indices'] = tuple(collated_batch['region_start_indices'])
        collated_batch['region_end_indices'] = tuple(collated_batch['region_end_indices'])

        # -- IMAGE_WIDTH -- IMAGE_HEIGHT -- BOXES
        collated_batch['image_height'] = None
        collated_batch['image_width'] = None
        collated_batch['boxes'] = None
        # No need to prepare features for ordering unless we have at least 2 real regions beside the empty region
        if 'image_width' in batch[0] and len(collated_batch['region_features']) >= 3:
            assert batch_size == 1, "BATCH_SIZE must be 1 for region ordering."

            collated_batch['image_height'] = batch[0]['image_height']
            collated_batch['image_width'] = batch[0]['image_width']

            # Get the spatial data and convert the coordinates into pixels instead of image ratio
            boxes = collated_batch['boxes'] = collated_batch['region_features'][1:, -5:-1].numpy().copy()
            boxes[:, (0, 2)] *= collated_batch['image_width']
            boxes[:, (1, 3)] *= collated_batch['image_height']
            collated_batch['boxes'] = boxes

        return collated_batch


class RawDataset(Dataset):
    """
    Loads all the relevant data from disk and prepares it for collating.
    Only the list of all example ids is permanently stored in memory.
    The RawDataset only loads the image, so the consumer needs to pass it through the visual network.

    Arguments:
        example_ids (list): a list of image ids
        image_dir (string): directory where the image files are located
        raw_dir (string): directory where the ground-truth region box coordinates are stored,
                          used for matching detections to the closest ground-truth boxes in ablation testing
    """

    def __init__(self, example_ids, image_dir, raw_dir=None):
        self.example_ids = example_ids
        self.image_dir = image_dir
        self.raw_dir = raw_dir

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        return self.load(self.example_ids[index])

    def get_example_id(self, index):
        return self.example_ids[index]

    def load(self, example_id):
        loaded_data = dict()
        loaded_data['example_id'] = example_id

        image_id = example_id
        load_iou_info = False
        if '_' in example_id:
            image_id = example_id.split('_')[0]
            load_iou_info = True

        # Load the raw image for this example
        image = cv2.imread(os_path.join(self.image_dir, image_id + '.jpg')).astype(dtype=np.float32)
        loaded_data['image'] = image

        if self.raw_dir is None:
            # Do not allow ground-truth boxes to be used
            loaded_data['boxes'] = None
        else:
            # Load the GT boxes for this example to be used for matching detections to the closest ground-truth boxes
            with open(os_path.join(self.raw_dir, image_id + '_raw.json'), 'rt') as raw_file:
                json_data = json.load(raw_file)

            boxes = [[bb['x_min'], bb['y_min'], bb['x_max'], bb['y_max']] for bb in json_data['all_bbs']]
            loaded_data['boxes'] = np.asarray(boxes)

            if load_iou_info:
                # Find the correct annotation for this example_id
                ann = None
                annotations = json_data['annotations']
                for i_ann in range(len(annotations)):
                    # The annotation number in the example_id won't always be the same as the i_ann because of filtering
                    if example_id == annotations[i_ann]['example_id']:
                        ann = annotations[i_ann]
                        break

                loaded_data['gt_entity_order'] = ann['entity_ids']  # list of entity_ids
                loaded_data['gt_entities'] = json_data['all_entities']  # dict entity_id -> list of bb_ids

        return loaded_data
