# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
from itertools import chain
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from preprocessing.preprocess_cv_features import box_relative_centers, box_areas


class RegionRankerNNIoU:
    def __init__(self, num_basic_features, num_hidden, num_feature_map_hidden, enable_training=False,
                 dropout_main=None, dropout_feauture_maps=None, learning_rate=0.00001, l2_weight=0.0, device="cpu:0"):
        self.device = device

        self.feature_map_layers = None
        self.main_layers = None
        self._build_network(num_basic_features, num_hidden, num_feature_map_hidden, dropout_main, dropout_feauture_maps)

        self.main_layers.to(device)
        if self.feature_map_layers is not None:
            self.feature_map_layers.to(device)

        if enable_training:
            self.main_layers.train()
            trainable_params = self.main_layers.parameters()

            if self.feature_map_layers is not None:
                self.feature_map_layers.train()
                trainable_params = chain(trainable_params, self.feature_map_layers.parameters())

            self.optimizer = torch.optim.Adam(params=trainable_params,
                                              lr=learning_rate, weight_decay=l2_weight)
            self.optimizer.zero_grad()

            self.loss_types = ["total_loss"]
        else:
            self.main_layers.eval()
            if self.feature_map_layers is not None:
                self.feature_map_layers.eval()

            self.optimizer = None

    def _build_network(self, num_basic_features, num_hidden, num_feature_map_hidden,
                       dropout_main, dropout_feauture_maps):
        if num_feature_map_hidden is None:
            self.feature_map_layers = None
        else:
            # Update the number of input features for the main layers
            num_basic_features += num_feature_map_hidden[-1]

            # Pass the feature maps through an initial mini-network
            feature_map_layers = list()
            num_inputs = [2048] + num_feature_map_hidden[:-1]  # the final hidden layer is our output from this mini-NN
            num_outputs = num_feature_map_hidden
            for i_layer in range(len(num_inputs)):
                i_str = str(i_layer)

                # Add dropout
                if dropout_feauture_maps is not None and dropout_feauture_maps[i_layer] > 0.0:
                    feature_map_layers.append(('dropout_' + i_str, nn.Dropout(p=dropout_feauture_maps[i_layer])))

                # Add the actual linear layer
                feature_map_layers.append(('linear_' + i_str,
                                           nn.Linear(in_features=num_inputs[i_layer], out_features=num_outputs[i_layer],
                                                     bias=True)))

                # Add non-linearity
                feature_map_layers.append(('relu_' + i_str, nn.ReLU()))

            # Turn these layers into a mini-network
            self.feature_map_layers = nn.Sequential(OrderedDict(feature_map_layers))

        # Build the main network
        num_inputs = [num_basic_features] + num_hidden
        num_outputs = num_hidden + [1]
        layers = list()
        for i_layer in range(len(num_inputs)):
            i_str = str(i_layer)

            # Add dropout
            if dropout_main is not None and dropout_main[i_layer] > 0.0:
                layers.append(('dropout_' + i_str, nn.Dropout(p=dropout_main[i_layer])))

            # Add the actual linear layer
            layers.append(('linear_' + i_str,
                           nn.Linear(in_features=num_inputs[i_layer], out_features=num_outputs[i_layer], bias=True)))

            # Add non-linearity
            if i_layer < (len(num_inputs)-1):
                layers.append(('relu_' + i_str, nn.LeakyReLU()))

        layers.append(('sigmoid_output', torch.nn.Sigmoid()))

        # Assemble the main network
        self.main_layers = nn.Sequential(OrderedDict(layers))

    def set_mode(self, mode):
        self.main_layers.train(mode == "train")
        if self.feature_map_layers is not None:
            self.feature_map_layers.train(mode == "train")

    def train(self, input_features, labels, update_weights=True):
        input_features = input_features.to(self.device)

        if self.feature_map_layers is None:
            input_features = input_features[:, :-2048]
        else:
            # Pass the feature map features through their mini-NN and replace them with its output
            feature_map_output = self.feature_map_layers(input_features[:, -2048:])
            input_features = torch.cat([input_features[:, :-2048], feature_map_output], dim=1)

        box_scores = self.main_layers(input_features).squeeze(1)
        loss = self._loss(scores=box_scores, labels=labels.to(self.device))

        if update_weights is True:
            # Backpropagate the loss
            loss.backward()

            # Run the optimizer step
            self.optimizer.step()

            # Reset the gradients
            self.optimizer.zero_grad()

        return {"total_loss": loss.detach().cpu().numpy()}

    def inference(self, input_features):
        box_features = self._prepare_inference_features(input_features)

        if self.feature_map_layers is not None:
            # Pass the feature map features through their mini-NN and replace them with its output
            feature_map_output = self.feature_map_layers(box_features[:, -2048:])
            box_features = torch.cat([box_features[:, :-2048], feature_map_output], dim=1)

        box_scores = self.main_layers(box_features).squeeze(1)
        _, highest_score_indices = torch.sort(box_scores.reshape(-1), descending=True)

        return highest_score_indices.cpu().numpy()

    def _prepare_inference_features(self, input_features):
        box_features = list()

        num_regions = len(input_features["categories"])

        # Flickr30k probabilities (box confidence)
        box_features.append(input_features["object_probs"])

        # Relative location to center x and y
        relative_center_x, relative_center_y = box_relative_centers(input_features["boxes"],
                                                                    image_half_width=input_features["image_width"] / 2,
                                                                    image_half_height=input_features["image_height"] / 2)
        box_features.append(relative_center_x)
        box_features.append(relative_center_y)

        # Relative area size to full image
        full_image_area = input_features["image_height"] * input_features["image_width"]
        relative_box_areas = box_areas(input_features["boxes"]) / full_image_area
        box_features.append(relative_box_areas)

        # Convert Flickr30k categories into one hot encodings
        one_hot_categories = np.zeros([num_regions, 7], dtype=np.float)
        one_hot_categories[list(range(num_regions)), input_features["categories"]] = 1.0
        box_features.append(one_hot_categories)

        # Feature maps for each box
        box_features.append(input_features["feature_maps"])

        # Stack all the features into a matrix with dimensions [num_boxes, num_features]
        box_features = torch.tensor(np.column_stack(box_features), dtype=torch.float32, device=self.device)

        return box_features

    # checkpoint_type indicates whether this was the current latest checkpoint or best validation/metric checkpoint, etc
    def save(self, checkpoint_dir, checkpoint_type, epoch):
        if self.feature_map_layers is not None:
            torch.save(self.feature_map_layers.state_dict(),
                       os_path.join(checkpoint_dir, checkpoint_type + '_feature_map_layers.pth'))
        torch.save(self.main_layers.state_dict(), os_path.join(checkpoint_dir, checkpoint_type + '_main_layers.pth'))
        torch.save(self.optimizer.state_dict(), os_path.join(checkpoint_dir, checkpoint_type + '_optimizer.pth'))
        with open(os_path.join(checkpoint_dir, checkpoint_type + '.txt'), 'at') as log:
            log.write(str(epoch) + '\n')

    def load(self, checkpoint_path, load_optimizer):
        self.main_layers.load_state_dict(torch.load(checkpoint_path + '_main_layers.pth'))

        try:
            self.feature_map_layers.load_state_dict(torch.load(checkpoint_path + '_feature_map_layers.pth'))
        except FileNotFoundError:
            print("No feature map layers were saved for this model.")

        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(checkpoint_path + '_optimizer.pth'))

        # Load the current epoch for this checkpoint
        with open(checkpoint_path + '.txt', 'rt') as log:
            epoch = int(log.read().splitlines()[-1])

        return epoch

    def _loss(self, scores, labels):
        return torch.nn.functional.binary_cross_entropy(scores, labels, reduction='mean')


class RegionRankerObjectProbs:
    def inference(self, box_features):
        scores = box_features["object_probs"]
        highest_prob_indices = np.flip(np.argsort(scores))

        # Flipped numpy arrays do not work well with pytorch tensors, so we need to make them contiguous
        return np.ascontiguousarray(highest_prob_indices)
