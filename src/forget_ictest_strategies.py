"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate subclass forgetting.
Seperate file to allow for easy reuse.
"""


import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset

from sklearn import linear_model, model_selection
from tqdm import tqdm

from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils import *
import ssd as ssd
import conf


def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[label].append((img, label, clabel))
    return classwise_ds


def build_retain_forget_sets(
    classwise_train, classwise_test, num_classes, forget_class
):
    # Getting the forget and retain validation data
    forget_valid = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_test[cls]:
                forget_valid.append((img, label, clabel))

    retain_valid = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_test[cls]:
                retain_valid.append((img, label, clabel))

    forget_train = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_train[cls]:
                forget_train.append((img, label, clabel))

    retain_train = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, label, clabel))

    return (retain_train, retain_valid, forget_train, forget_valid)


def replace_indexes(logger, dataset: torch.utils.data.Dataset, indexes: Union[List[int], np.ndarray], only_mark: bool = False):
    if not only_mark:
        ''' Golatkar's replacement
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(list(set(range(len(dataset))) - set(indexes)), size=len(indexes))
        dataset.data[indexes] = dataset.data[new_indexes]
        dataset.targets[indexes] = dataset.targets[new_indexes]
        '''
        old_size = len(dataset.data)
        retain_indexes = list(set(range(len(dataset))) - set(indexes))
        logger.debug(f'Removed targets: {dataset.targets[indexes]}')
        dataset.data = dataset.data[retain_indexes]
        dataset.targets = dataset.targets[retain_indexes]
        new_size = len(dataset.data)
        logger.debug(f'Oldsize: {old_size}\t Newsize:{new_size}')
        return dataset
    else:
        # Notice the -1 to make class 0 work
        logger.debug(f'Marking forget set targets: {dataset.targets[indexes]}')
        dataset.targets[indexes] = - dataset.targets[indexes] - 1
        return dataset


def add_confusion(dataset: torch.utils.data.Dataset, confusion: List[List[int]], conf_copy: bool):
    num_classes = len(confusion)
    confset = deepcopy(dataset)
    confused_indices = []
    for i in range(num_classes):
        class_i_indexes = np.flatnonzero(np.array(dataset.targets) == i)
        last_rep = 0
        for j in range(num_classes):
            if i == j:
                continue
            # Replace first C_{i, j} starting from last_rep indexes
            for itr in range(confusion[i][j]):
                idx = class_i_indexes[last_rep + itr]
                confset.targets[idx] = j
                confused_indices.append(idx)
            last_rep += confusion[i][j]

    return confset, confused_indices, dataset


def replace_class(logger, dataset: torch.utils.data.Dataset, class_to_replace: int, num_indexes_to_replace: int = None,
                  seed: int = 0, only_mark: bool = False):
    indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)  # random number generator
        indexes = rng.choice(
            indexes, size=num_indexes_to_replace, replace=False)
        logger.info(f"Replacing indexes {indexes}")
    return replace_indexes(logger, dataset, indexes, only_mark)


def get_loaders(logger, classwise_train, classwise_test, class_to_replace: int = None, num_indexes_to_replace: int = None, seed: int = 1,
                only_mark: bool = False, root: str = None, batch_size=128, shuffle=True, **dataset_kwargs):
    '''

    :param dataset_name: Name of dataset to use
    :param class_to_replace: If not None, specifies which class to replace completely or partially
    :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
                                   `num_indexes_to_replace` samples
    :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
                               indexes_to_replace can be specidied.
    :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
                 always in the same order
    :param root: Root directory to initialize the dataset
    :param batch_size: Batch size of data loader
    :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
    :param dataset_kwargs: Extra arguments to pass to the dataset init.
    :return: The train_loader and test_loader
    '''
    seed_everything(seed)

    if root is None:
        root = os.path.expanduser('~/data')
    train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    valid_set = deepcopy(train_set)
    rng = np.random.RandomState(seed)

    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(rng.choice(class_idx, int(
            validsplit*len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)

    train_idx = list(set(range(len(train_set)))-set(valid_idx))

    train_set_copy = deepcopy(train_set)

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    '''Modified here by Shash'''
    confused_indices = None

    if class_to_replace is not None:
        logger.info(
            f'C_f: {class_to_replace}\t N_f: {num_indexes_to_replace}\t only_mark: {only_mark}\t train_size: {len(train_set.data)}')
        if class_to_replace == -1:
            rng = np.random.RandomState(seed-1)
            indexes = rng.choice(range(len(train_set)),
                                 size=num_indexes_to_replace, replace=False)
            train_set = replace_indexes(logger, train_set, indexes, only_mark)
        else:
            train_set = replace_class(logger, train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace,
                                      seed=seed-1, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader, valid_loader, test_loader, confused_indices


def get_metric_scores(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
):
    loss_acc_dict = evaluate(model, valid_dl, device)
    retain_acc_dict = evaluate(model, retain_valid_dl, device)
    zrf = UnLearningScore(model, unlearning_teacher,
                          forget_valid_dl, 128, device)
    d_f = evaluate(model, forget_valid_dl, device)
    mia = get_membership_attack_prob(
        retain_train_dl, forget_train_dl, valid_dl, model)

    return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], zrf, mia, d_f["Acc"])


def baseline(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def retrain(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dataset_name,
    model_name,
    device,
    **kwargs,
):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    if model_name == "ViT":
        epochs = getattr(conf, f"{dataset_name}_{model_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_{model_name}_MILESTONES")
    else:
        epochs = getattr(conf, f"{dataset_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_MILESTONES")
    _ = fit_one_cycle(
        epochs,
        model,
        retain_train_dl,
        retain_valid_dl,
        milestones=milestones,
        device=device,
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def finetune(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    _ = fit_one_cycle(
        5, model, retain_train_dl, retain_valid_dl, lr=0.02, device=device
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

# Catastrophic Forgetting in the last k layers, picked from Goel et al., Evaluating Inexact Unlearning


def freeze_first_k_layers(model, k):
    # Get model layers
    layer_names = list(model._modules.keys())
    if k > len(layer_names):
        raise ValueError(
            f"Cannot freeze {k} layers; the model has only {len(layer_names)} layers."
        )

    layer_count = 0
    for layer in model.children():
        # Freeze only the first k layers
        if layer_count < k:
            for param in layer.parameters():
                param.requires_grad = False
            print(
                f"Layer {layer_count+1} ({layer.__class__.__name__}) frozen.")
        layer_count += 1

    return model


def CF_k(model, unlearning_teacher, retain_train_dl, retain_valid_dl, forget_train_dl, forget_valid_dl, valid_dl, device, **kwargs):
    k = 2
    if 'k' in kwargs:
        k = kwargs['k']
    model.to(device)
    modules = list(model.children())

    model = freeze_first_k_layers(model, len(modules) - k)

    _ = fit_one_cycle(
        5, model, retain_train_dl, retain_valid_dl, lr=0.02, device=device
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

# Exact Unlearning in the last k layers, picked from Goel et al., Evaluating Inexact Unlearning


def initialize_last_k_layers(model, k):
    # Check for valid k
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Get model layers
    layer_names = list(model._modules.keys())

    if k > len(layer_names):
        raise ValueError(
            f"Cannot replace {k} layers; the model has only {len(layer_names)} layers."
        )

    # Replace layers starting from the end
    for i in range(1, k + 1):
        layer_name = layer_names[-i]
        layer = getattr(model, layer_name)

        if isinstance(layer, nn.Linear):
            # Initialize the last fully connected layer
            setattr(model, layer_name, nn.Linear(
                layer.in_features, layer.out_features))
            nn.init.kaiming_uniform_(
                getattr(model, layer_name).weight, nonlinearity="relu"
            )
            if getattr(model, layer_name).bias is not None:
                nn.init.constant_(getattr(model, layer_name).bias, 0)
        elif isinstance(layer, nn.Conv2d):
            # Initialize convolutional layers
            setattr(
                model,
                layer_name,
                nn.Conv2d(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                ),
            )
            nn.init.kaiming_uniform_(
                getattr(model, layer_name).weight, nonlinearity="relu"
            )
            if getattr(model, layer_name).bias is not None:
                nn.init.constant_(getattr(model, layer_name).bias, 0)
        elif isinstance(layer, nn.Sequential):
            # Initialize layers inside a sequential module
            for block in layer:
                for submodule in block.children():
                    if isinstance(submodule, nn.Conv2d):
                        nn.init.kaiming_uniform_(
                            submodule.weight, nonlinearity="relu")
                        if submodule.bias is not None:
                            nn.init.constant_(submodule.bias, 0)
                        break  # Only modify one Conv2d layer per block
                break  # Only modify one block per Sequential module


def EU_k(model, unlearning_teacher, retain_train_dl, retain_valid_dl, forget_train_dl, forget_valid_dl, valid_dl, device, **kwargs):
    k = 2
    if 'k' in kwargs:
        k = kwargs['k']
    modules = list(model.children())
    model = freeze_first_k_layers(model,  len(modules) - k)
    initialize_last_k_layers(model=model, k=k)
    model.to(device)

    fit_one_cycle(
        5, model, retain_train_dl, retain_valid_dl, lr=0.02, device=device
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def blindspot(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = random.sample(
        retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset))
    )

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        retain_data=retain_train_subset,
        forget_data=forget_train_dl.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
    )

    return get_metric_scores(
        student_model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_superclasses,
    forget_superclass,
    device,
    **kwargs,
):
    unlearninglabels = list(range(num_superclasses))
    unlearning_trainset = []

    unlearninglabels.remove(forget_superclass)

    for x, y, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, y, random.choice(unlearninglabels)))

    for x, y, clabel in retain_train_dl.dataset:
        unlearning_trainset.append((x, y, clabel))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    _ = fit_one_unlearning_cycle(
        3, model, unlearning_train_set_dl, retain_valid_dl, device=device, lr=0.0001
    )
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_superclasses,
    device,
    **kwargs,
):
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, _, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target ==
                                       target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_superclasses:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_train_dl.dataset, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def UNSIR(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_subclasses,
    forget_subclass,
    forget_superclass,
    device,
    **kwargs,
):
    classwise_train = get_classwise_ds(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
        num_subclasses,
    )
    noise_batch_size = 32
    retain_valid_dl = DataLoader(
        retain_valid_dl.dataset, batch_size=noise_batch_size)
    # collect some samples from each class
    num_samples = 500
    retain_samples = []
    for i in range(num_subclasses):
        if i != forget_subclass:
            retain_samples += classwise_train[i][:num_samples]

    forget_class_label = forget_superclass
    img_shape = next(iter(retain_train_dl.dataset))[0].shape[-1]
    noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
    noise = UNSIR_noise_train(
        noise, model, forget_class_label, 250, noise_batch_size, device=device
    )
    noisy_loader = UNSIR_create_noisy_loader(
        noise, forget_class_label, retain_samples, noise_batch_size, device=device
    )
    # impair step
    _ = fit_one_unlearning_cycle(
        1, model, noisy_loader, retain_valid_dl, device=device, lr=0.0001
    )
    # repair step
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (
                retain_samples[i][0].cpu(),
                torch.tensor(retain_samples[i][2]),
                torch.tensor(retain_samples[i][2]),
            )
        )

    heal_loader = torch.utils.data.DataLoader(
        other_samples, batch_size=128, shuffle=True
    )
    _ = fit_one_unlearning_cycle(
        1, model, heal_loader, retain_valid_dl, device=device, lr=0.0001
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def pdr_tuning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = pdr.calc_importance(forget_train_dl)
    original_importances = pdr.calc_importance(full_train_dl)
    pdr.modify_weight(original_importances, sample_importances)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )
