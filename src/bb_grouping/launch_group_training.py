# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from bb_grouping.bb_group_models import GroupingNN
from os import makedirs, path as os_path
import json
import numpy as np
import torch
import matplotlib


# Prevent errors due to lack of a connected display
matplotlib.use('Agg')
# ---------------------------------------------------------
import matplotlib.pyplot as plt


DEVICE = "cpu:0"
RNG = np.random.default_rng(1234)
# Based on the ratio of 1.0 label instances (~49.8) and 0.0 label instance (~50.2) to down-weigh the more common one
LABEL_0_WEIGHT = 0.9937868789176668  # 49.8441879333235 / 50.1558120666765


def run_epoch(model, dataset, batch_size, update_weights):
    batch_losses = {loss_type: np.asarray([0], dtype=np.float64) for loss_type in model.loss_types}

    num_examples = len(dataset)

    if (batch_size is None) or (batch_size > num_examples):
        batch_size = num_examples
        num_batches = 1
    else:
        num_batches = int((num_examples / batch_size) + 0.5)

    if update_weights and (batch_size > 1):
        # Shuffle the order if we're training
        example_order = RNG.permutation(num_examples)
    else:
        example_order = range(num_examples)

    for i_batch in range(num_batches):
        start = i_batch * batch_size
        end = min((i_batch + 1) * batch_size, num_examples)
        current_batch_size = end - start

        current_examples = example_order[start:end]
        features = torch.tensor(dataset[current_examples, :-1], dtype=torch.float32, device=DEVICE)
        labels = torch.tensor(dataset[current_examples, -1], dtype=torch.float32, device=DEVICE)

        # Down-weigh the more common class based on its prevalence in the whole train dataset
        example_weights = torch.ones([current_batch_size], dtype=torch.float32, device=DEVICE)
        example_weights[labels == 0.0] = LABEL_0_WEIGHT

        # Further down-weigh all examples in this batch based on its size
        if current_batch_size < batch_size:
            example_weights = example_weights * (current_batch_size/batch_size)

        losses = model.train(input_features=features,
                             labels=labels,
                             example_weights=example_weights,
                             update_weights=update_weights)

        for loss_type in batch_losses:
            batch_losses[loss_type] += losses[loss_type]

    return {loss_type: batch_losses[loss_type] / num_examples for loss_type in batch_losses}


def train(model, datasets, checkpoint_dir, batch_size=None, starting_epoch=0, num_epochs=200, eval_frequency=1):
    loss_log = {split + '_' + loss_type: list() for split in ["train", "val"] for loss_type in model.loss_types}
    loss_log["epoch"] = list()

    best_loss = None

    for epoch in range(starting_epoch, starting_epoch + num_epochs):
        print("TRAINING EPOCH", epoch)

        model.set_mode("train")
        losses = run_epoch(model, datasets["train"], batch_size, update_weights=True)

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
            losses = run_epoch(model, datasets["val"], batch_size, update_weights=False)
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

    datasets = dict()
    for split in ["train", "val"]:
        datasets[split] = np.load("../data/group_features_" + split + ".npy")

    learning_rate = 0.01
    l2_weight = 0.0
    num_features = 8
    num_hidden = [16, 8]
    dropout = [0.0, 0.0, 0.0]
    batch_size = None
    num_epochs = 200
    checkpoint_dir = '../checkpoints/bb_group_model'

    print("checkpoint_dir", checkpoint_dir)
    print("num_hidden", num_hidden)
    print("learning_rate", learning_rate)
    print("l2_weight", l2_weight)
    print("dropout", dropout)
    print("batch_size", batch_size)
    print("num_epochs", num_epochs)

    makedirs(checkpoint_dir, exist_ok=True)

    # Store network settings
    outpath = os_path.join(checkpoint_dir, "settings.json")
    with open(outpath, 'wt') as outfile:
        json.dump({"num_features": num_features, "num_hidden": num_hidden, "dropout": dropout}, outfile)
    group_model = GroupingNN(num_features, num_hidden, enable_training=True,
                             learning_rate=learning_rate, l2_weight=l2_weight, dropout=dropout, device=DEVICE)

    train(model=group_model, datasets=datasets, checkpoint_dir=checkpoint_dir,
          batch_size=batch_size, starting_epoch=0, num_epochs=num_epochs, eval_frequency=1)

    plot_learning_curves(output_path=os_path.join(checkpoint_dir, 'loss_curves.png'),
                         loss_path=os_path.join(checkpoint_dir, 'loss_log.npz'))
