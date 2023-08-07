# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from region_ranker.model import RegionRankerNNIoU
from region_ranker.dataloaders import RegionSelectionIoULoader
from os import makedirs, path as os_path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib


# Prevent errors due to lack of a connected display
matplotlib.use('Agg')
# ---------------------------------------------------------
import matplotlib.pyplot as plt


_DEVICE = "cpu:0"
_NUM_WORKERS = 3
_HARD_THRESHOLD = 0.4


def run_epoch(model, dataloader, update_weights):
    batch_losses = {loss_type: np.asarray([0], dtype=np.float64) for loss_type in model.loss_types}
    for batch in dataloader:
        if "example_weights" in batch:
            losses = model.train(input_features=batch["features"],
                                 labels=batch["labels"],
                                 example_weights=batch["example_weights"],
                                 update_weights=update_weights)
        else:
            losses = model.train(input_features=batch["features"],
                                 labels=batch["labels"],
                                 update_weights=update_weights)

        for loss_type in batch_losses:
            batch_losses[loss_type] += losses[loss_type]

    return {loss_type: batch_losses[loss_type] / len(dataloader.dataset) for loss_type in batch_losses}


def train(model, dataloaders, checkpoint_dir, starting_epoch=0, num_epochs=30, eval_frequency=1):
    loss_log = {split + '_' + loss_type: list() for split in ["train", "val"] for loss_type in model.loss_types}
    loss_log["epoch"] = list()

    best_loss = None

    for epoch in range(starting_epoch, starting_epoch + num_epochs):
        print("TRAINING EPOCH", epoch)

        model.set_mode("train")
        losses = run_epoch(model, dataloaders["train"], update_weights=True)

        if epoch % eval_frequency == 0:
            # Log current epoch
            loss_log["epoch"].append(epoch)

            # Log training losses
            for loss_type in losses:
                loss_log["train_" + loss_type].append(losses[loss_type])

            print("Saving latest checkpoint...")
            model.save(checkpoint_dir=checkpoint_dir, checkpoint_type="latest", epoch=epoch)
            print("Checkpoint saved!")

            print("Running validation epoch...")
            model.set_mode("eval")
            losses = run_epoch(model, dataloaders["val"], update_weights=False)
            print("Validation epoch finished!")
            if (best_loss is None) or (losses["total_loss"] < best_loss):
                best_loss = losses["total_loss"]

                print("Saving total_loss checkpoint...")
                model.save(checkpoint_dir=checkpoint_dir, checkpoint_type="total_loss", epoch=epoch)
                print("Checkpoint saved!")

            # Log validation losses
            for loss_type in losses:
                loss_log["val_" + loss_type].append(losses[loss_type])

            print("Current train and val loss", loss_log["train_total_loss"][-1], loss_log["val_total_loss"][-1])

    print("Best loss", best_loss)

    loss_log = {k: np.stack(loss_log[k]) for k in loss_log}
    np.savez(os_path.join(checkpoint_dir, "loss_log.npz"), **loss_log)

    plot_learning_curves(output_path=os_path.join(checkpoint_dir, "loss_curves.png"), loss_data=loss_log)


def setup_dataloaders(nn_type, data_dir, splits_path, splits, hard_threshold=None):
    with open(splits_path, 'rt') as splits_file:
        splits_data = json.load(splits_file)['splits']

    dataloaders = dict()
    for split in splits:
        if nn_type == "iou":
            dataset = RegionSelectionIoULoader(splits_data[split], data_dir, hard_threshold)
        else:
            assert False, "Unkown nn_type " + str(nn_type)

        print("Found {} examples out of {} requested for split {}.".format(len(dataset), len(splits_data[split]), split))
        dataloaders[split] = DataLoader(dataset, batch_size=1, shuffle=(split == "train"), num_workers=_NUM_WORKERS,
                                        collate_fn=lambda batch: batch[0])  # return the first (and only) example

    return dataloaders


def plot_learning_curves(output_path, loss_data=None, loss_path=None):
    assert (loss_data is not None and loss_path is None) or \
           (loss_data is None and loss_path is not None), "Please provide EITHER loss_data OR loss_path."

    if loss_path is not None:
        loss_data = dict(np.load(loss_path))

    epochs = loss_data.pop("epoch")

    loss_types = list(loss_data.keys())
    loss_types.sort()
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.rainbow(np.linspace(0, 1, len(loss_types)))))
    for loss_type in loss_types:
        if loss_type[:3] == "val":
            linestyle = '--'
        else:
            linestyle = '-'

        plt.plot(epochs, loss_data[loss_type], linestyle=linestyle, label=loss_type)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Loss Curves")
    plt.ylim(top=min(loss_data["train_total_loss"][0], loss_data["train_total_loss"][2]*4))
    plt.savefig(
        fname=output_path,
        dpi='figure',
        bbox_inches='tight'
    )
    plt.close()


if __name__ == "__main__":
    torch.random.manual_seed(333)

    dataloaders = setup_dataloaders(nn_type="iou",
                                    data_dir='../data/region_selector/features_iou/',
                                    splits_path='../data/splits_full.json',
                                    splits=['train', 'val'],
                                    hard_threshold=_HARD_THRESHOLD)

    settings = {
        "num_basic_features": 11,
        "num_hidden": [30, 20],
        "num_feature_map_hidden": [20],
        "dropout_main": None,
        "dropout_feauture_maps": None,
        "learning_rate": 0.00001,
        "l2_weight": 0.0,
    }
    num_epochs = 30
    checkpoint_dir = '../checkpoints/region_ranking'

    makedirs(checkpoint_dir, exist_ok=True)

    # Store network settings
    outpath = os_path.join(checkpoint_dir, "settings.json")
    with open(outpath, 'wt') as outfile:
        json.dump(settings, outfile)

    print("checkpoint_dir", checkpoint_dir)
    print(settings)

    regsel_model = RegionRankerNNIoU(**settings, enable_training=True, device=_DEVICE)

    train(model=regsel_model, dataloaders=dataloaders, checkpoint_dir=checkpoint_dir,
          starting_epoch=0, num_epochs=num_epochs, eval_frequency=1)

    plot_learning_curves(output_path=os_path.join(checkpoint_dir, 'loss_curves.png'),
                         loss_path=os_path.join(checkpoint_dir, 'loss_log.npz'))
