"""
From https://github.com/vikram2000b/bad-teaching-unlearning / https://arxiv.org/abs/2205.08096
"""

import itertools
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from utils import AverageMeter, get_outputs, get_error


def JSDiv(p, q):
    m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)


# ZRF/UnLearningScore https://arxiv.org/abs/2205.08096
def UnLearningScore(tmodel, gold_model, forget_dl, batch_size, device):
    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for batch in forget_dl:
            x, y, cy = batch
            x = x.to(device)
            model_output = tmodel(x)
            gold_model_output = gold_model(x)
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(
                F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    return 1 - JSDiv(model_preds, gold_model_preds)


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False
    )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device)
                     for tensor in batch]
            data, _, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()


@torch.no_grad()
def actv_dist(model1, model2, dataloader, device="cuda"):
    sftmx = torch.nn.Softmax(dim=1)
    distances = []
    for batch in dataloader:
        x, _, _ = batch
        x = x.to(device)
        model1_out = model1(x)
        model2_out = model2(x)
        diff = torch.sqrt(
            torch.sum(
                torch.square(
                    F.softmax(model1_out, dim=1) - F.softmax(model2_out, dim=1)
                ),
                axis=1,
            )
        )
        diff = diff.detach().cpu()
        distances.append(diff)
    distances = torch.cat(distances, axis=0)
    return distances.mean()


# Interclass Confusion Test (IC Test) from https://github.com/shash42/Evaluating-Inexact-Unlearning/tree/master
@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds, all_targets = torch.tensor([]), torch.tensor([])
    all_preds, all_targets = all_preds.to(device), all_targets.to(device)
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        model.eval()
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_targets = torch.cat((all_targets, labels), dim=0)
    return all_preds, all_targets


def classpairs_sorted_by_conf(cm):
    '''
    Returns list of 5-tuple sorted in ascending order of total confusion:
    (cm[i, j] + cm[j, i], i, j, cm[i, j], cm[j, i])
    '''
    pair_confs = []
    for i in range(cm.shape[0]):
        for j in range(i+1, cm.shape[1]):
            pair_confs.append((cm[i, j] + cm[j, i], i, j, cm[i, j], cm[j, i]))
    pair_confs.sort(reverse=True)
    return pair_confs


def gen_confu_mat(model, loader, device, title='Confusion Matrix', path=None):
    preds, targets = get_all_preds(model, loader, device)
    stacked = torch.stack((targets, preds.argmax(dim=1)), dim=1)
    cmt = torch.zeros(preds.shape[1], preds.shape[1], dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        tl, pl = int(tl), int(pl)
        cmt[tl, pl] = cmt[tl, pl] + 1
    cm = np.array(cmt)

    if cm.shape[0] >= 5:
        pair_confs = classpairs_sorted_by_conf(cm)
        with open(f"{path}.txt", "w") as f:
            for pair in pair_confs:
                print(
                    f'Classes ({pair[1]}, {pair[2]}) - {pair[3]} + {pair[4]} = {pair[0]}', file=f)
            for i in range(cm.shape[0]):
                print(f'Class {i} correct: {cm[i, i]}', file=f)
        return cm

    cmap = plt.cm.Blues
    classes = [i for i in range(preds.shape[1])]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if path is not None:
        plt.savefig(f"{path}.png")
    plt.close()
    return cm


def CRT_score(cm, forget_class):
    FP = cm[forget_class][forget_class]
    FN = 0
    for i in range(cm.shape[0]):
        if i == forget_class:
            continue
        FN += cm[i][forget_class]
    return FP, FN


def Conf_Score(cm, classes):
    confscore = 0
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            ci, cj = classes[i], classes[j]
            confscore += cm[ci][cj] + cm[cj][ci]
    return confscore


def acc_from_cmtx(cm):
    corr, tot = 0, 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tot += cm[i][j]
            if i == j:
                corr += cm[i][j]
    return float(corr)/tot


def get_metrics(model, dataloader, criterion):
    activations, predictions, sample_info = get_outputs(model, dataloader)
    losses, errors = AverageMeter(), AverageMeter()

    for inputs, outputs, labels in sample_info:
        loss = criterion(outputs, labels)
        losses.update(loss.item(), n=inputs.size(0))
        errors.update(get_error(outputs, labels), n=inputs.size(0))

    return losses.avg, errors.avg, activations, predictions


def activations_predictions(model, dataloader, name):
    criterion = torch.nn.CrossEntropyLoss()
    losses, errors, activations, predictions = get_metrics(
        model, dataloader, criterion)
    print(f"{name} -> Loss:{np.round(losses,3)}, Error:{errors}")
    return activations, predictions


def ICTest(model, device, savepath, retain_loader, forget_loader, test_loader_full,
           name, forget_class, exch_classes, MIA_info=None, test_loader_r=None, test_loader_f=None):
    activations_r, predictions_r = activations_predictions(
        model, retain_loader, f'{name}_Model_D_r')
    activations_f, predictions_f = activations_predictions(
        model, forget_loader, f'{name}_Model_D_f')
    activations_t, predictions_t = activations_predictions(
        model, test_loader_full, f'{name}_Model_D_t')

    if test_loader_r is not None:
        activations_te_r, predictions_te_r = activations_predictions(
            model, test_loader_r, f'{name}_Model_D_te_r')
    if test_loader_f is not None:
        activations_te_f, predictions_te_f = activations_predictions(
            model, test_loader_f, f'{name}_Model_D_te_f')

    CRTRes, ConfRes = {}, {}
    cm_t = gen_confu_mat(model, test_loader_full, device,
                         f"{name} - Test - Confusion Matrix", f"{savepath}/{name}-test")
    cm_f = gen_confu_mat(model, forget_loader, device,
                         f"{name} - Forget - Confusion Matrix", f"{savepath}/{name}-forget")
    cm_r = gen_confu_mat(model, retain_loader, device,
                         f"{name} - Retain - Confusion Matrix", f"{savepath}/{name}-retain")
    cm_te_r, cm_te_f = None, None

    if test_loader_r is not None:
        cm_te_r = gen_confu_mat(model, test_loader_r, device,
                                f"{name} - Test_r - Confusion Matrix", f"{savepath}/{name}-test_r")
    if test_loader_f is not None:
        cm_te_f = gen_confu_mat(model, test_loader_f, device,
                                f"{name} - Test_f - Confusion Matrix", f"{savepath}/{name}-test_f")

    if forget_class is not None and forget_class != -1:
        print(f'Class Forgetting performance of {name}')
        CRTRes["t_FP"], CRTRes["t_FN"] = CRT_score(cm_t, forget_class)
        print(f'Test:\tt_FP = {CRTRes["t_FP"]}\tt_FN = {CRTRes["t_FN"]}')
        CRTRes["r_FP"], CRTRes["r_FN"] = CRT_score(cm_r, forget_class)
        print(f'Retain:\tr_FP = {CRTRes["r_FP"]}\tr_FN = {CRTRes["r_FN"]}')
        CRTRes["f_FP"], CRTRes["f_FN"] = CRT_score(cm_f, forget_class)
        print(f'Forget:\tf_FP = {CRTRes["f_FP"]}\tf_FN = {CRTRes["f_FN"]}')

        if cm_te_r is not None:
            CRTRes["te_r_FP"], CRTRes["te_r_FN"] = CRT_score(
                cm_te_r, forget_class)
            print(
                f'Test_r:\tte_r_FP = {CRTRes["te_r_FP"]}\tte_r_FN = {CRTRes["te_r_FN"]}')
        if cm_te_f is not None:
            CRTRes["te_f_FP"], CRTRes["te_f_FN"] = CRT_score(
                cm_te_f, forget_class)
            print(
                f'Test_f:\tte_f_FP = {CRTRes["te_f_FP"]}\tte_f_FN = {CRTRes["te_f_FN"]}')

    print(f'Exch classes - {exch_classes}')
    if exch_classes is not None:
        print(f'Confusion Forgetting performance of {name}')
        ConfRes["t_Conf"] = Conf_Score(cm_t, exch_classes)
        print(f'Test:\tConf_Score = {ConfRes["t_Conf"]}')
        ConfRes["f_Conf"] = Conf_Score(cm_f, exch_classes)
        print(f'Forget:\tConf_Score = {ConfRes["f_Conf"]}')
        ConfRes["r_Conf"] = Conf_Score(cm_r, exch_classes)
        print(f'Retain:\tConf_Score = {ConfRes["r_Conf"]}')
