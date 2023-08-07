# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from ordering.sinkhorn import build_feature_matrix


class SinkhornDataset(Dataset):
    def __init__(self, example_ids, data_dir, embeddings_path, max_num_regions=10, category_type="vg",
                 vg_to_flickr30k_path=None):
        assert category_type in ["vg", "entities_top3"]
        self.category_type = category_type
        if self.category_type == "entities_top3":
            assert vg_to_flickr30k_path is not None, \
                "category_type=='entities_top3' requires that vg_to_flickr30k_path is not None"

            # Load the data to translate the vg categories into Flickr30k categories if needed
            with open(vg_to_flickr30k_path) as infile:
                vg_to_flickr30k_data = json.load(infile)
                self.vg_idx_to_flickr30k_idx = vg_to_flickr30k_data["vg_idx_to_flickr30k_idx"]

        self.class_embeddings = torch.load(embeddings_path)

        self.max_num_regions = max_num_regions
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

        if self.category_type == "vg":
            categories = torch.tensor(data["object_classes"][:, 0])
        elif self.category_type == "entities_top3":
            # Re-calculate the object_probs based on the flickr30k classes of the top 3 probs
            categories = list(map(lambda i_box:
                                  self._entities_top3_category(data["object_probs"][i_box, :],
                                                               data["object_classes"][i_box, :]),
                                  range(len(data["object_probs"]))))

            categories = torch.tensor(categories)
        else:
            assert False, "Unknown category type:" + self.category_type

        feature_maps = torch.tensor(data["region_features"])
        spatial_features = torch.tensor(data["spatial_features"])

        input_features = build_feature_matrix(categories, feature_maps, spatial_features, self.max_num_regions,
                                              self.class_embeddings)

        num_regions = min(len(feature_maps), self.max_num_regions)

        return {'example_id': example_id, 'input_features': input_features, 'num_regions': num_regions}

    def _entities_top3_category(self, sorted_probs, sorted_classes):
        top_flickr_classes = [self.vg_idx_to_flickr30k_idx[i_class] for i_class in sorted_classes]
        unique_classes = list(set(top_flickr_classes))

        if len(unique_classes) == 1:
            top_flickr_category = top_flickr_classes[0]

            return top_flickr_category

        top_classes = np.asarray(top_flickr_classes)
        flickr_probs = [sorted_probs[top_classes == current_class].sum() for current_class in unique_classes]
        top_unique_idx = np.asarray(flickr_probs).argmax()

        top_flickr_category = unique_classes[top_unique_idx]

        return top_flickr_category


class CollateSinkhornTraining:
    """
    Takes a batch of data example dicts and returns a single dict where each entry is batched.
    The collated batch contains the following entries:
        example_ids         - list of example ids in this batch [batch_size]
        ordered_features    - torch matrix of concatenated region embeddings [batch_size, max_num_regions, 300+2048+4]
                              (in the same order they came in from each un-collated example)
        unordered_features  - torch matrix of concatenated region embeddings [batch_size, max_num_regions, 300+2048+4]
                              (ordered by random permutations)
        padding_mask        - torch matrix length-mask for each example [batch_size, max_num_regions]
    """
    def __call__(self, batch):
        batch_size = len(batch)

        collated_batch = dict()

        # -- EXAMPLE_ID
        # Turn into a single list
        collated_batch['example_ids'] = [batch[i]['example_id'] for i in range(batch_size)]

        # -- ORDERED FEATURES
        # Gather all and stack into a single torch matrix
        collated_batch['ordered_features'] = torch.cat([batch[i]['input_features'] for i in range(batch_size)],
                                                       dim=0).detach()

        # -- UNORDERED FEATURES
        max_num_regions = collated_batch['ordered_features'].size(1)
        permutations = [torch.randperm(max_num_regions) for _ in range(batch_size)]

        collated_batch['unordered_features'] = torch.cat([batch[i]['input_features'][:, permutations[i], :]
                                                          for i in range(batch_size)], dim=0).detach()

        # -- PADDING MASK
        padding_masks = [torch.cat([torch.ones([1, batch[i]['num_regions']]),
                                   torch.zeros([1, max_num_regions-batch[i]['num_regions']])], dim=1)
                         for i in range(batch_size)]
        collated_batch['padding_mask'] = torch.cat(padding_masks, dim=0)

        return collated_batch
