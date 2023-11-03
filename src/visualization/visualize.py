import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.patches as mpatches
from skimage.exposure import adjust_log

# ### small boiler plate to add src to sys path
import sys
from pathlib import Path

file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------

from src.visualization.confusion_matrix_pretty_print import pretty_plot_confusion_matrix


def show(func):
    """
    Helper decorator for immediate showing of plots
    Parameters
    ----------
    func: function
        Must returns figure (matplotlib.pyplot).
    Returns
    -------

    """

    def wrapper(*args, **kwargs):
        fig = func(*args, **kwargs)
        plt.show()

    return wrapper


@show
def visualize_bands(field_4D, time=0, from_x=5, to_x=10, from_y=5, to_y=10, what='unknown'):
    """Visualizes bands' (on x-axis) values (on y-axis) for all pixels within specified slice.
    Plots the spectral profile.
    Parameters
    ----------
    field_4D : np.ndarray (TIME x BANDS x X x Y) 
    time : int
        date index
    from_x, to_x, from_y, to_y : int
        specification of slice
    """
    num_bands = field_4D.shape[1]
    fig, ax = plt.subplots()
    plt.plot(('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12'),
             field_4D[time, [0, 1, 2, 3, 4, 5, 8, 6, 7],
             from_y:to_y, from_x:to_x].reshape((-1, (to_x - from_x) * (to_y - from_y))) / 10000)
    ax.set_xticklabels(('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12'))
    ax.set_xlabel('Spectarl band')  # ax.set_xlabel('Spektrálne pásmo')
    ax.set_ylabel('Reflectance')  # ax.set_ylabel('Odrazivosť')
    ax.set_title(f'Spectral profile: {what}')  # ax.set_title(f'Spektrálny profil: {what}')
    return fig


@show
def visualize_time(field_4D, band=2, from_x=5, to_x=10, from_y=5, to_y=10):
    """See visualize_bands(field_4D, ...)
    """
    num_times = field_4D.shape[0]
    fig, ax = plt.subplots()
    plt.plot(field_4D[:, band, from_x:to_x, from_y:to_y].reshape((num_times, (to_x - from_x) * (to_y - from_y))))
    ax.set_title(f'Band: {band} in time')
    return fig


@show
def plot_mask(mask_type, path_tile):
    """
    Plots given mask over given Sentinel 2 L2A data.
    Parameters
    ----------
    mask_type : str
        can be 'CLOUDS', 'SNOW', 'CLASSIFICATION'
    path_tile : str
        relative path to tile ending with .SAFE which mask will be displayed 

    Returns
    -------

    """

    from src.helpers.sentinel import sentinel_load_clouds

    mask = sentinel_load_clouds(path_tile, mask_type=mask_type)

    if mask.size > 0:
        fig, ax = plt.subplots()
        ax.imshow(mask[0])
        ax.set_title(f'Mask: {mask_type}')
        return fig
    else:
        print('Wrong parameters have been passed')


def plot_learning_history(val_metrics, train_metrics):
    """
    Used for SentiNet.
    Plots metrics in epoch.
    Similar to `plot_cnn_history`
    Parameters
    ------------
    train_metrics: list
    val_metrics: list
        List of dictionaries each containing metrics.
        Length of list is number of epochs.

    Returns
    -------

    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    val_accuracy = [metric['val_accuracy'] for metric in val_metrics]
    train_accuracy = [metric['train_accuracy'] for metric in train_metrics]
    val_loss = [metric['val_loss'] for metric in val_metrics]
    train_loss = [metric['train_loss'] for metric in train_metrics]
    # val_avg_recall = [metric['val_avg_recall'] for metric in val_metrics]
    # train_avg_recall = [metric['train_avg_recall'] for metric in train_metrics]
    val_jaccard = [metric['val_jaccard'] for metric in val_metrics]
    train_jaccard = [metric['train_jaccard'] for metric in train_metrics]
    val_avg_dice = [metric['val_avg_dice_coeff'] for metric in val_metrics]
    train_avg_dice = [metric['train_avg_dice_coeff'] for metric in train_metrics]

    ax1.plot(train_loss)
    ax1.plot(val_loss, linestyle='dashed')

    ax1.set_title('Loss function of model')  # ax1.set_title('Účelová funkcia modelu')
    ax1.set_ylabel('Value')  # ax1.set_ylabel('Hodnota')
    ax1.set_xlabel('Epoch')  # ax1.set_xlabel('Epocha')
    ax1.legend(['train', 'val'], loc='upper right')

    ax2.plot(train_accuracy)
    ax2.plot(val_accuracy, linestyle='dashed')
    # ax2.plot(train_avg_recall)
    # ax2.plot(val_avg_recall, linestyle='dashed')
    ax2.plot(train_jaccard)
    ax2.plot(val_jaccard, linestyle='dashed')
    ax2.plot(train_avg_dice)
    ax2.plot(val_avg_dice, linestyle='dashed')

    ax2.set_title('Metrics')  # ax2.set_title('Metriky modelu')
    ax2.set_ylabel('Value')  # ax2.set_ylabel('Hodnota')
    ax2.set_xlabel('Epoch')  # ax2.set_xlabel('Epocha')
    ax2.legend(['train acc', 'val acc',
                # 'train recall',
                # 'val recall',
                'train jaccard', 'val jaccard', 'train dice',
                'val dice'], loc='upper left')
    plt.tight_layout()

    return fig


def plot_metrics_from_csv(csv_path, add_epoch):
    """
    Auxiliary function to read specific metrics from csv file written by CSVLogger
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.CSVLogger.html
    Parameters
    ----------
    csv_path
    add_epoch: int
        const value to add to epoch
    Returns
    -------

    """
    metrics = pd.read_csv(csv_path)

    def iter_aux(prefix, df):
        metrics = {}
        p = ['epoch', f'{prefix}_loss', f'{prefix}_accuracy', f'{prefix}_jaccard',
             f'{prefix}_avg_dice_coeff']
        for i, z in enumerate(zip(df, p)):
            if i > 0:
                metrics[z[1]] = df[z[0]].to_list()
        adjusted = []
        for i in range(len(metrics[f'{prefix}_loss'])):
            adjusted.append({k: v for k, v in zip(p[1:],
                                                  [metrics[k][i] for k in p[1:]])})
        return adjusted

    v = metrics.iloc[:][['epoch', 'val_loss_epoch', 'val_Accuracy', 'val_JaccardIndex', 'val_F1Score']]
    v.dropna(axis=0, how='any', inplace=True,
             subset=['val_loss_epoch', 'val_Accuracy', 'val_JaccardIndex', 'val_F1Score'])

    val_metrics = iter_aux('val', v)

    t = metrics.iloc[:][['epoch', 'train_loss_epoch', 'train_Accuracy', 'train_JaccardIndex', 'train_F1Score']]
    t.dropna(axis=0, how='any', inplace=True,
             subset=['train_loss_epoch', 'train_Accuracy', 'train_JaccardIndex', 'train_F1Score'])

    train_metrics = iter_aux('train', t)

    return plot_learning_history(val_metrics, train_metrics)


def plot_cnn_counts(y, y_test_pred, class_map):
    """Compares the counts of crops in the dataset vs. counts of predicted crops
    Parameters
    ---------
    y : np.ndarray
        Integer encoded target name in train dataset.
    y_test_pred : np.ndarray
        Integer encoded target name in test dataset.
    class_map : dict
        Dictionary where keys are target names and values are theirs integer encoded representation.
        E.g. {'building': 3}
    """
    inv_map = {v: k for k, v in class_map.items()}
    y = np.argmax(y, axis=1)
    y = [inv_map[val] for val in y]
    y = pd.DataFrame(data=np.array(y), columns=['class'])
    pred_y = np.argmax(y_test_pred, axis=1)
    pred_y = [inv_map[val] for val in pred_y]
    pred_y = pd.DataFrame(data=np.array(pred_y), columns=['class'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    sns.countplot(data=y, ax=ax1, order=class_map.keys(), x="class")
    ax1.set_title('Distribution (actual in test dataset)')  # ax1.set_title('Rozloženie (skutočné v test. datasete)')
    ax1.set_xlabel('Class')  # ax1.set_xlabel('Trieda')
    ax1.set_ylabel('Count')  # ax1.set_ylabel('Počet')
    sns.countplot(data=pred_y, ax=ax2, order=class_map.keys(), x="class")
    ax2.set_title(
        'Distribution (predicted in test dataset)')  # ax2.set_title('Rozloženie (predikované v test. datasete)')
    ax2.set_xlabel('Class')  # ax2.set_xlabel('Trieda')
    ax2.set_ylabel('Count')  # ax2.set_ylabel('Počet')

    return fig


def plot_dataset_class_distribution(y, class_map, counts=None):
    if counts is None:
        y = np.argmax(y, axis=1)
        counts = np.bincount(y)
    labels = list(class_map.keys())
    theme = plt.get_cmap('tab20')

    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", [theme(1. * i / len(labels))
                                for i in range(len(labels))])

    ax.pie(counts, shadow=False, startangle=90, wedgeprops={'linewidth': 3, "edgecolor": "k"})
    ax.set_title('Distribution of classes')
    ax.legend(labels=labels, loc="best")
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig


def plot_cnn_confusion_matrix(confusion_matrix, classes, normalize=False, scientific=False, fz=12,
                              annot=True,
                              cmap='Blues', print_IoU=False, title='Confusion matrix', figsize=[25, 25],
                              ignore_index=[]):
    """
    Plots the confusion matrix.
    Parameters
    ----------
    figsize: List[int]
        Defines size of resulting figure.
    title: str
        Specifiy title used in plot.
    print_IoU: bool
        Whether to plot IoU (intersection over union) metric.
    cmap: str
        Color map
    normalize: bool
        Whether to normalize confusion matrix
    confusion_matrix : np.ndarray
        output of sklearn.confusion_matrix
    classes : list
        List of classes used in predictions.
    ignore_index: list
        List of indices to be ignored when calculating metrics
    """

    # THANKS TO https://github.com/wcipriano/pretty-print-confusion-matrix

    cm = pd.DataFrame(data=confusion_matrix, columns=classes, index=classes)
    fig = pretty_plot_confusion_matrix(df_cm=cm, annot=annot, cmap=cmap, pred_val_axis='x', figsize=figsize,
                                       print_IoU=print_IoU, title=title, fz=fz, normalize=normalize,
                                       ignore_index=ignore_index, scientific_notation=scientific)
    return fig


def plot_log_cm(cm, labels, encoded_labels):
    """
    Plots log-scale confusion matrix. Good for very big confusion matrices with very large
    cell values.
    Parameters
    ----------
    cm: 2d array
    labels: list of labels (string representation of labels)
    encoded_labels: list of integer codes for labels

    Returns
    -------
    returns matplotlib.pyplot.Figure object
    """
    fig, ax = plt.subplots(figsize=(11, 11))
    log_cm = np.log(cm + 1)
    ax = sns.heatmap(log_cm, cmap="plasma", xticklabels=encoded_labels, yticklabels=labels, ax=ax)
    ax.tick_params(axis='y', labelrotation=45, labelsize=10)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.yaxis.set_label_position("right")

    return fig


def plot_cm(cm, labels, encoded_labels, normalize=True, cmap='plasma'):
    """
    Plots normalized confusion matrix over true labels.
    Parameters
    ----------
    cm: 2d array
    labels: list of labels (string representation of labels)
    encoded_labels: list of integer codes for labels
    normalize: whether to normalize

    Returns
    -------
    returns matplotlib.pyplot.Figure object
    """
    fig, ax = plt.subplots(figsize=(11, 11))
    if normalize:
        cm2 = cm / cm.sum(axis=1)[:, None]
        cm2[np.where(cm.sum(axis=1) == 0)[0]] = 0.
    else:
        cm2 = cm
    cmm = np.round(cm2, decimals=2)
    ax = sns.heatmap(cm2, cmap=cmap, xticklabels=encoded_labels, yticklabels=labels, ax=ax, annot=cmm)
    ax.tick_params(axis='y', labelrotation=45, labelsize=10)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.yaxis.set_label_position("right")

    return fig


def plot_proba_mask(proba_mask):
    """
    Plot probability mask
    Parameters
    ----------
    proba_mask: np.ndarray

    Returns
    -------
    returns matplotlib.pyplot.Figure object
    """
    fig, ax = plt.subplots()
    sns.heatmap(proba_mask, xticklabels=False, yticklabels=False, annot=False, ax=ax)
    ax.set_title(f'Confidence map')

    return fig


def plot_lulc(data, labels, cmap, reclass_rules=None):
    """
    Plots segmentation mask
    Parameters
    ----------
    data: np.ndarray
        2D array [H x W]
        with class codes
    labels: list
        Labels of classes where position of label should correspond with class code i.e.
        if background is encoded with 0 label for background  should be on 0th position in list
    cmap: dict
        Dictionary where keys are class codes and values are hexadecimal colors
    Returns
    -------
    returns matplotlib.pyplot.Figure object
    """
    if reclass_rules is not None:
        uniq = np.unique(data)
        cmap = {i: cmap[reclass_rules[i]] for i in range(uniq[-1] + 1)}
        labels = [labels[reclass_rules[i]] for i in range(uniq[-1] + 1)]
    patches = [mpatches.Patch(color=cmap[i], label=label) for i, label in enumerate(labels)]
    un = np.unique(data)
    kk = []
    pp = []
    for i, j in enumerate(labels):
        if j in pp or i not in un:
            continue
        kk.append(i)
        pp.append(j)
    pat = [patches[i] for i in np.intersect1d(un, np.array(kk))]

    fig, ax = plt.subplots(figsize=(9, 6))
    mask_show = np.array([[cmap[i] for i in j] for j in data])
    ax.imshow(mask_show)
    ax.legend(handles=pat, bbox_to_anchor=(1.00, 1), loc=2, borderaxespad=0.)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    return fig


def plot_rgb(data):
    """
    Plots rgb image
    Parameters
    ----------
    data: np.ndarray
        3D array [C x H x W]
        where C (channel) dimension has Red, Green, Blue channels in this order
    Returns
    -------
    returns matplotlib.pyplot.Figure object
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # equalize histogram
    # data = equalize_hist(data)
    # apply log correction to contrast
    # img = adjust_log(data, 0.06).transpose(1, 2, 0)
    img = data.transpose(1, 2, 0)
    ax.imshow(img / np.quantile(img, 0.99))
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    return fig


def plot_ndvi(ndvi: np.ndarray):
    """
    plot ndvi spectral index
     Parameters
    ----------
    ndvi: np.ndarray
        2D array [H x W]
        representing ndvi index i.e. normalized in (-1, 1) interval
    Returns
    -------
    returns matplotlib.pyplot.Figure object
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(ndvi.astype(np.float32), cmap='RdYlGn')
    fig.colorbar(im, orientation='vertical')
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)

    return fig

# TODO get some inspiration -> https://datavizproject.com/
