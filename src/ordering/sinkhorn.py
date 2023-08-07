# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
import munkres
import torch
from ordering.sinkhorn_network import SinkhornNet


# Builds a matrix of size [1, self.max_num_regions, 300+2048+4]
def build_feature_matrix(categories, feature_maps, spatial_features, max_num_regions, class_embeddings, verbose=False):
    categories = class_embeddings[categories, :]
    network_features = torch.cat((categories, feature_maps, spatial_features,), dim=1)

    num_regions = len(network_features)
    if num_regions > max_num_regions:
        if verbose:
            print("WARNING: Skipping {} out of {} regions.".format(num_regions - max_num_regions, num_regions))

        network_features = network_features[:max_num_regions, :]
    elif num_regions < max_num_regions:
        network_features = network_features
        padding = torch.zeros([max_num_regions - num_regions, network_features.size(1)]).to(
            class_embeddings.device)
        network_features = torch.cat([network_features, padding], dim=0)

    # Unsqueeze to add the batch dimension
    return network_features.unsqueeze(0)


def prepare_sinkhorn_features(region_features, selected_boxes_idx):
    categories = region_features["categories"][selected_boxes_idx]
    feature_maps = torch.tensor(region_features["feature_maps"][selected_boxes_idx])
    boxes = region_features["boxes"][selected_boxes_idx]
    image_width = region_features["image_width"]
    image_height = region_features["image_height"]

    rel_center_x = torch.tensor((boxes[:, 0:1] + boxes[:, 2:3]) * 0.5, dtype=torch.float32) / image_width
    rel_center_y = torch.tensor((boxes[:, 1:2] + boxes[:, 3:4]) * 0.5, dtype=torch.float32) / image_height

    rel_length_x = torch.tensor(boxes[:, 2:3] - boxes[:, 0:1], dtype=torch.float32) / image_width
    rel_length_y = torch.tensor(boxes[:, 3:4] - boxes[:, 1:2], dtype=torch.float32) / image_height

    spatial_features = torch.cat([rel_center_x, rel_center_y, rel_length_x, rel_length_y], dim=1)

    return categories, feature_maps, spatial_features


class SinkhornOrdering:
    def __init__(self, max_num_regions=10, num_iterations=20, tau=0.1, device="cpu:0",
                 training_params=None, class_embeddings_path=None):
        self.model = SinkhornNet(max_num_regions, num_iterations, tau).to(device)
        self.max_num_regions = max_num_regions
        self.device = device

        if training_params is None:
            self.model.eval()
            self.optimizer = None
        else:
            self.model.train()
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=training_params["learning_rate"],
                                              weight_decay=training_params["l2_weight"])
            self.optimizer.zero_grad()

        self.munkres = munkres.Munkres()

        self.class_embeddings = None
        if class_embeddings_path is not None:
            self.class_embeddings = torch.load(class_embeddings_path)

    def train(self, ordered_features, unordered_features, padding_mask, update_weights=True):
        sinkhorn_order = self.model.forward(unordered_features)

        # Attempt to reconstruct the original input
        reconstructed_input = torch.matmul(sinkhorn_order.transpose(1, 2), unordered_features)

        # Calculate the loss except for the padded rows
        loss = torch.nn.functional.mse_loss(input=reconstructed_input,
                                            target=ordered_features,
                                            reduction='none')
        loss = loss[padding_mask.to(dtype=torch.bool)].sum() / padding_mask.sum()

        if update_weights:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach()

    def __call__(self, region_features, selected_boxes_idx):
        return self.inference(region_features, selected_boxes_idx)

    def inference(self, region_features, selected_boxes_idx):
        num_regions = min(len(selected_boxes_idx), self.max_num_regions)
        if num_regions == 1:
            return [0]

        categories, feature_maps, spatial_features = prepare_sinkhorn_features(region_features,
                                                                               selected_boxes_idx)

        sinkhorn_features = build_feature_matrix(categories=categories,
                                                 feature_maps=feature_maps,
                                                 spatial_features=spatial_features,
                                                 max_num_regions=self.max_num_regions,
                                                 class_embeddings=self.class_embeddings)

        sinkhorn_order = self.model.forward(sinkhorn_features.to(self.device))
        sinkhorn_order = sinkhorn_order.squeeze(0).transpose(0, 1).detach().cpu().numpy()

        # Zero out the padded regions
        if num_regions < self.max_num_regions:
            num_padded_regions = self.max_num_regions - num_regions
            sinkhorn_order[:, -num_padded_regions:] = 0.0

        # Send in the negative of the matrix since munkres will look for lowest rather than highest values
        rows_cols_ordered = self.munkres.compute(-sinkhorn_order)

        # The ordered columns show which order all the feature columns should be selected from
        # (the rows should always be 0, 1, ..., n - otherwise we would need to order the cols first)
        ordered_idx = [col for row, col in rows_cols_ordered if col < num_regions]

        return ordered_idx

    # Builds a matrix of size [self.max_num_regions, 300+2048+4]
    def _prepare_input(self, categories, feature_maps, spatial_features):
        categories = self.class_embeddings[categories, :]
        network_features = torch.cat((categories, feature_maps, spatial_features,), dim=1)

        num_regions = len(network_features)
        if num_regions > self.max_num_regions:
            print("WARNING: Skipping {} out of {} regions.".format(num_regions-self.max_num_regions, num_regions))
            network_features = network_features[:self.max_num_regions, :]
        elif num_regions < self.max_num_regions:
            network_features = network_features
            padding = torch.zeros([self.max_num_regions-num_regions, network_features.size(1)]).to(self.class_embeddings.device)
            network_features = torch.cat([network_features, padding], dim=0)

        # For now, our batch size is always 1
        return network_features.unsqueeze(0)

    # checkpoint_type indicates whether this was the current latest checkpoint or best validation/metric checkpoint, etc
    def save(self, checkpoint_dir, checkpoint_type, epoch):
        torch.save(self.model.state_dict(), os_path.join(checkpoint_dir, checkpoint_type + '_model.pth'))
        torch.save(self.optimizer.state_dict(), os_path.join(checkpoint_dir, checkpoint_type + '_optimizer.pth'))
        with open(os_path.join(checkpoint_dir, checkpoint_type + '.txt'), 'at') as log:
            log.write(str(epoch) + '\n')

    def load(self, checkpoint_path, load_optimizer):
        self.model.load_state_dict(torch.load(checkpoint_path + '_model.pth'))
        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(checkpoint_path + '_optimizer.pth'))
        # Load the current epoch for this checkpoint
        with open(checkpoint_path + '.txt', 'rt') as log:
            epoch = int(log.read().splitlines()[-1])

        return epoch
