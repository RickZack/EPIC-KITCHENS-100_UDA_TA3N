import itertools
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import csv

def randSelectBatch(input, num):
    id_all = torch.randperm(input.size(0)).cuda()
    id = id_all[:num]
    return id, input[id]

def plot_confusion_matrix(path, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    num_classlabels = cm.sum(axis=1) # count the number of true labels for all the classes
    np.putmask(num_classlabels, num_classlabels == 0, 1) # avoid zero division

    if normalize:
        cm = cm.astype('float') / num_classlabels[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(13, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    factor = 100 if normalize else 1
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*factor, fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path)

def save_result_csv(filename, modality, temp_aggr, use_target, seqex, rna_weight, prec1_v, prec_1n, prec_1a,
                    prec_5v, prec_5n, prec5_5a):
    file_exist = os.path.isfile(filename)
    modalities = '-'.join(modality)
    with open(filename, 'a') as f:
        writer = csv.writer(f)

        if not file_exist:
            writer.writerow(['Modality', 'Temporal Aggregation', 'Precision@1_verb',
                            'Precision@1_noun', 'Precision@1_action', 'Precision@5_verb',
                            'Precision@5_noun', 'Precision@5_action'])
        scenario = 'source only'
        if rna_weight > 0:
            scenario = 'UDA' if use_target != 'none' else 'DG'
        layer = 'SqEx' if seqex else 'Linear'
        additional_layer = layer if rna_weight > 0 else ''
        t_aggr = f"{temp_aggr} {scenario} {additional_layer}"
        if rna_weight > 0:
            t_aggr += f' RNA (w={rna_weight})'

        writer.writerow([modalities, t_aggr, prec1_v, prec_1n, prec_1a,
                            prec_5v, prec_5n, prec5_5a])
