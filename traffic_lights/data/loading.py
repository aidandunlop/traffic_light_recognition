import torch
from .constants import CLASS_LABEL_MAP
from .sampler import StratifiedSampler
from traffic_lights.lib import engine_utils


def split_data(full_dataset):
    labels_per_image = full_dataset.annotations_groupby_frame.apply(
        lambda x: x["Annotation tag"].unique()
    )
    # find combinations of traffic lights
    label_combinations = [
        [CLASS_LABEL_MAP[label] for label in labels] for labels in labels_per_image
    ]

    maxLen = max(map(len, label_combinations))
    for row in label_combinations:
        while len(row) < maxLen:
            row.append(0)

    # perform first split of dataset into training and testing
    training_indices, testing_indices = list(
        StratifiedSampler(torch.tensor(label_combinations), 1, test_size=0.3)
    )

    # perform second split of training set into training and validation set
    training_labels = [label_combinations[index] for index in training_indices]

    training_indices, validation_indices = list(
        StratifiedSampler(torch.tensor(training_labels), 1, test_size=0.3)
    )

    # TO DO: if smoke test, take tiny sample
    training_dataset = torch.utils.data.Subset(full_dataset, training_indices)
    testing_dataset = torch.utils.data.Subset(full_dataset, testing_indices)
    validation_dataset = torch.utils.data.Subset(full_dataset, validation_indices)

    print("Subset sizes")
    print(
        "Training: {}, Validation: {}, Testing: {}".format(
            len(training_dataset), len(validation_dataset), len(testing_dataset)
        )
    )

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=engine_utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=engine_utils.collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=engine_utils.collate_fn,
    )

    return (data_loader_train, data_loader_test, data_loader_val)
