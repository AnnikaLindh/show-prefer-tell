# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from preprocessing.preprocess_cv_features import calculate_group_pair_features, box_areas


class GroupingNN:
    def __init__(self, num_features, num_hidden, confidence_threshold=0.5, enable_training=False,
                 learning_rate=0.01, l2_weight=0.0, dropout=None, device="cpu:0", gather_stats=False):
        self.device = device
        self.confidence_threshold = confidence_threshold
        if gather_stats:
            self.all_scores = list()
        else:
            self.all_scores = None

        self.network = None
        self._build_network(num_features, num_hidden, dropout)
        self.network.to(device)

        if enable_training:
            self.network.train()
            self.optimizer = torch.optim.Adam(params=self.network.parameters(),
                                              lr=learning_rate, weight_decay=l2_weight)
            self.optimizer.zero_grad()

            self.loss_types = ["total_loss"]
        else:
            self.network.eval()
            self.optimizer = None

    def _build_network(self, num_features, num_hidden, dropout):
        num_inputs = [num_features] + num_hidden
        num_outputs = num_hidden + [1]

        layers = list()
        for i_layer in range(len(num_inputs)):
            i_str = str(i_layer)
            if dropout is not None and dropout[i_layer] > 0.0:
                layers.append(('dropout_' + i_str, nn.Dropout(p=dropout[i_layer])))
            layers.append(('linear_' + i_str,
                           nn.Linear(in_features=num_inputs[i_layer], out_features=num_outputs[i_layer], bias=True)))
            if i_layer < (len(num_inputs)-1):
                layers.append(('relu_' + i_str, nn.ReLU()))

        layers.append(('sigmoid_output', torch.nn.Sigmoid()))

        self.network = nn.Sequential(OrderedDict(layers))

    def set_mode(self, mode="train"):
        self.network.train(mode == "train")

    def train(self, input_features, labels, example_weights, update_weights=True):
        box_scores = self.network(input_features.to(self.device)).squeeze(1)
        loss = self._loss(scores=box_scores, labels=labels.to(self.device),
                          example_weights=example_weights.to(self.device))

        if update_weights is True:
            # Backpropagate the loss
            loss.backward()

            # Run the optimizer step
            self.optimizer.step()

            # Reset the gradients
            self.optimizer.zero_grad()

        return {"total_loss": loss.detach().cpu().numpy()}

    def __call__(self, bb_features):
        return self.inference(bb_features)

    def inference(self, input_features):
        # Prepare the features for all potential pairs
        features, pairs = self._prepare_inference_features(input_features)

        if features is None:
            return list()

        # Calculate the likelihood of each potential pair
        pair_scores = self.network(features)

        if self.all_scores is not None:
            self.all_scores.extend(pair_scores.flatten().tolist())

        # Select the pairs (list of 2 indices each) where the score is above a threshold
        accepted_idx = torch.where(pair_scores > self.confidence_threshold)[0]
        accepted_pairs = [pairs[idx] for idx in accepted_idx]
        if len(accepted_pairs) == 0:
            return list()

        # Merge pairs that share indices
        grouped_entities = list()
        num_grouped = 0
        for pair in accepted_pairs:
            grouped_entities, num_grouped = merge_pair(pair[0], pair[1], grouped_entities, num_grouped)

        return grouped_entities

    def _prepare_inference_features(self, input_features):
        boxes = input_features["boxes"]
        categories = input_features["categories"]

        unique_categories = np.unique(categories)
        # Remove categories 5 and 6 (clothing and bodyparts)
        unique_categories = unique_categories[unique_categories < 5]
        if len(unique_categories) == len(categories):
            return None, None

        # Calculate the x-centers and y-centers of the bbs
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

        # Calculate the area of each bb
        all_areas = box_areas(boxes)

        # Add the features for each pair in each entity type category
        all_features = list()
        all_pairs = list()
        for cat in unique_categories:
            current_idx = np.where(categories == cat)[0]
            current_boxes = boxes[current_idx]
            if len(current_boxes) < 2:
                continue

            current_centers_x = centers_x[current_idx]
            current_centers_y = centers_y[current_idx]
            current_areas = all_areas[current_idx]

            x_order = np.argsort(current_centers_x, axis=None)
            prepared_features = calculate_group_pair_features(order=x_order,
                                                              current_bbs=current_boxes,
                                                              centers_x=current_centers_x,
                                                              centers_y=current_centers_y,
                                                              bb_areas=current_areas,
                                                              category=cat)
            # Store the indices into the full list of bbs for each of the bbs in each pair
            all_pairs.extend([[current_idx[x_order[i]], current_idx[x_order[i+1]]] for i in range(len(x_order)-1)])

            y_order = np.argsort(current_centers_y, axis=None)
            new_pairs = [i for i in range(len(y_order) - 1)
                         if ((x_order[i] != y_order[i]) or (x_order[i + 1] != y_order[i + 1]))]
            if len(new_pairs) > 0:
                additional_features = calculate_group_pair_features(order=y_order,
                                                                    current_bbs=current_boxes,
                                                                    centers_x=current_centers_x,
                                                                    centers_y=current_centers_y,
                                                                    bb_areas=current_areas,
                                                                    category=cat)
                prepared_features = np.concatenate([prepared_features, additional_features[new_pairs]])

                # Store the indices into the full list of bbs for each of the bbs in each pair
                all_pairs.extend([[current_idx[y_order[i]], current_idx[y_order[i+1]]]
                                  for i in range(len(y_order) - 1) if i in new_pairs])

            # Add the current entity type's features
            all_features.append(prepared_features)

        if len(all_features) == 0:
            return None, None

        all_features = torch.tensor(np.concatenate(all_features, axis=0), dtype=torch.float32, device=self.device)

        return all_features, all_pairs

    # checkpoint_type indicates whether this was the current latest checkpoint or best validation/metric checkpoint, etc
    def save(self, checkpoint_dir, checkpoint_type, epoch):
        torch.save(self.network.state_dict(), os_path.join(checkpoint_dir, checkpoint_type + '_model.pth'))
        torch.save(self.optimizer.state_dict(), os_path.join(checkpoint_dir, checkpoint_type + '_optimizer.pth'))
        with open(os_path.join(checkpoint_dir, checkpoint_type + '.txt'), 'at') as log:
            log.write(str(epoch) + '\n')

    def load(self, checkpoint_path, load_optimizer):
        self.network.load_state_dict(torch.load(checkpoint_path + '_model.pth'))
        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(checkpoint_path + '_optimizer.pth'))
        # Load the current epoch for this checkpoint
        with open(checkpoint_path + '.txt', 'rt') as log:
            epoch = int(log.read().splitlines()[-1])

        return epoch

    def _loss(self, scores, labels, example_weights):
        return torch.nn.functional.binary_cross_entropy(scores, labels, weight=example_weights, reduction='mean')


def merge_pair(first_bb, second_bb, grouped_entities, num_grouped):
    first_group = [idx for idx in range(len(grouped_entities)) if first_bb in grouped_entities[idx]]
    second_group = [idx for idx in range(len(grouped_entities)) if second_bb in grouped_entities[idx]]

    if len(first_group) > 0:
        if len(second_group) > 0:
            if first_group[0] != second_group[0]:
                # Both are in different groups already, so merge those groups
                grouped_entities[first_group[0]].extend(grouped_entities[second_group[0]])
                del grouped_entities[second_group[0]]
        else:
            # The first bb is already in a group, so add the second bb to that group
            grouped_entities[first_group[0]].append(second_bb)
            num_grouped += 1
    elif len(second_group) > 0:
        # The second bb is already in a group, so add the first bb to that group
        grouped_entities[second_group[0]].append(first_bb)
        num_grouped += 1
    else:
        # Neither are already in a grouped entity, so create a new one
        grouped_entities.append([first_bb, second_bb])
        num_grouped += 2

    return grouped_entities, num_grouped
