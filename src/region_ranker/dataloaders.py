# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import listdir, path as os_path
import numpy as np
import torch
from torch.utils.data import Dataset


class RegionSelectionIoULoader(Dataset):
    def __init__(self, example_ids, data_dir, hard_threshold=None):
        self.data_dir = data_dir
        self.hard_threshold = hard_threshold

        # Find the examples that are valid for our Region Selection training
        examples_files = listdir(self.data_dir)
        self.example_ids = [example_id for example_id in example_ids if example_id + '_entity_selection.npz' in examples_files]

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        return self.load(self.example_ids[index])

    def get_example_id(self, index):
        return self.example_ids[index]

    def load(self, example_id):
        example_data = np.load(os_path.join(self.data_dir, str(example_id) + '_entity_selection.npz'))

        features = torch.tensor(example_data["features"], dtype=torch.float32)
        labels = torch.tensor(example_data["labels"], dtype=torch.float32)
        if self.hard_threshold is not None:
            labels[labels < self.hard_threshold] = 0.0
            labels[labels >= self.hard_threshold] = 1.0

        return {"example_id": example_id, "features": features, "labels": labels}
