# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
from os import makedirs
import json
import numpy as np
from ordering.dataloaders import SinkhornDataset, CollateSinkhornTraining
from ordering.sinkhorn import SinkhornOrdering
import torch
from torch.utils.data import DataLoader
import matplotlib


# Prevent errors due to lack of a connected display
matplotlib.use('Agg')
# ---------------------------------------------------------
import matplotlib.pyplot as plt


_DEVICE = "cpu:0"
_EVAL_FREQ = 1


def plot_learning_curves(output_path, loss_data=None, loss_path=None):
    assert (loss_data is not None and loss_path is None) or \
           (loss_data is None and loss_path is not None), "Please provide EITHER loss_data OR loss_path."

    if loss_path is not None:
        loss_data = dict(np.load(loss_path))

    epochs = loss_data.pop("epochs")

    plt.plot(epochs, loss_data["train_losses"], label="training loss")
    plt.plot(epochs, loss_data["val_losses"], label="validation loss")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.title("Loss Curves")
    plt.savefig(
        fname=output_path,
        dpi='figure',
        bbox_inches='tight'
    )
    plt.close()


def train(model, dataloaders, num_epochs, checkpoint_dir):
    epochs = list()
    train_losses = list()
    val_losses = list()
    best_val = None

    for epoch in range(num_epochs):
        print("TRAINING EPOCH", epoch)

        train_loss = run_epoch(model, dataloaders['train'], update_weights=True)

        if epoch % _EVAL_FREQ == 0:
            print("Saving latest checkpoint...")
            model.save(checkpoint_dir=checkpoint_dir, checkpoint_type="latest", epoch=epoch)
            print("Checkpoint saved!")

            print("Running validation epoch...")
            val_loss = run_epoch(model, dataloaders['val'], update_weights=False)
            print("Validation epoch finished!")
            if (best_val is None) or (val_loss < best_val):
                model.save(checkpoint_dir=checkpoint_dir, checkpoint_type="val", epoch=epoch)
                print("Checkpoint saved!")
                best_val = val_loss

            # Log losses
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print("Current train and val loss", train_loss, val_loss)

    loss_log = {"epochs": epochs, "train_losses": train_losses, "val_losses": val_losses}
    np.savez(os_path.join(checkpoint_dir, "loss_log.npz"), **loss_log)

    plot_learning_curves(output_path=os_path.join(checkpoint_dir, "loss_curves.png"), loss_data=loss_log)


def run_epoch(model, dataloader, update_weights=False):
    epoch_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch_loss = model.train(ordered_features=batch["ordered_features"].to(_DEVICE),
                                 unordered_features=batch["unordered_features"].to(_DEVICE),
                                 padding_mask=batch["padding_mask"].to(_DEVICE),
                                 update_weights=update_weights)

        epoch_loss += batch_loss
        num_batches += 1

    return epoch_loss / num_batches


def setup_dataloaders(data_dir, splits_path, splits, embeddings_path, category_type, vg_to_flickr30k_path=None):
    with open(splits_path, 'rt') as splits_file:
        splits_data = json.load(splits_file)['splits']

    dataloaders = dict()
    for split in splits:
        dataset = SinkhornDataset(example_ids=splits_data[split], data_dir=data_dir, embeddings_path=embeddings_path,
                                  category_type=category_type, vg_to_flickr30k_path=vg_to_flickr30k_path)
        dataloaders[split] = DataLoader(dataset, batch_size=10, shuffle=(split == "train"), num_workers=7,
                                        collate_fn=CollateSinkhornTraining())

    return dataloaders


if __name__ == "__main__":
    torch.random.manual_seed(333)

    data_dir = "../data/region_order/sinkhorn"
    sinkhorn_splits_path = "../data/region_order/sinkhorn_splits.json"
    category_type = "entities_top3"
    vg_to_flickr30k_path = '../data/models/vg_flickr30ksplit/flickr30k_category_info.json'
    embeddings_path = "../data/region_order/entities_glove_6B_300d_vocab_embeddings.pt"
    training_params = {"learning_rate": 1e-05, "l2_weight": 0.0}
    num_epochs = 5
    checkpoint_dir = "../checkpoints/sinkhorn_model"

    makedirs(checkpoint_dir, exist_ok=True)

    # Setup the dataloader
    dataloaders = setup_dataloaders(data_dir=data_dir, splits_path=sinkhorn_splits_path, splits=['train', 'val'],
                                    embeddings_path=embeddings_path, category_type=category_type,
                                    vg_to_flickr30k_path=vg_to_flickr30k_path)

    # Create an instance of SinkhornOrdering
    sinkhorn = SinkhornOrdering(device=_DEVICE, training_params=training_params)

    train(model=sinkhorn, dataloaders=dataloaders, num_epochs=num_epochs, checkpoint_dir=checkpoint_dir)
