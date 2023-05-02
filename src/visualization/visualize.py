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
def plot_index(name, func, path_tile):
    """
    Plots given index over given Sentinel 2 tile.
    Parameters
    ----------
    name : str
        name of index e.g. 'ndvi'
    func : function
        represents index from indices.indices.py
    path_tile : str
        relative path to tile ending with .SAFE on which will be calculated index
    Returns
    -------

    """
    index = func(path_tile, name=name, export=False)
    fig, ax = plt.subplots()
    im = ax.imshow(index[0], cmap='gist_earth')
    fig.colorbar(im, orientation='vertical')
    ax.set_title(name)
    return fig


@show
def plot_band(channel, band, path_tile):
    """
    Plots given band in channel over given Sentinel 2 tile.
    Channel and band must corresponds.
    Parameters
    ----------
    channel : str
        indicating resolution e.g. 'R20m'
    band : str
        indicating band e.g. 'B02'
    path_tile : str 
        relative path to tile ending with .SAFE which band will be displayed

    Returns
    -------

    """
    from src.helpers.sentinel import sentinel_load_channel

    band_ = sentinel_load_channel(path_tile, channel=channel, band=band)

    if band_.size > 0:
        fig, ax = plt.subplots()
        ax.imshow(band_[0])
        ax.set_title(band)
        return fig
    else:
        print("No such channel or band.")


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


@show
def plot_polarisation(path_tile, polarisation):
    """
    Plots given polarisation of given Sentinel 1 SAR data (with enhanced contrast, plotted values are squared).
    Parameters
    ----------
    path_tile : str
        relative path to Sentinel 1 tile ending with .SAFE
    polarisation : str
        can be 'VV', 'VH', 'HH', 'HV'
    Returns
    -------

    """
    from src.helpers.sentinel import sentinel_load_measurement
    polar = sentinel_load_measurement(path_tile, polarisation=polarisation)

    if polar.size > 0:
        fig, ax = plt.subplots()
        ax.imshow(np.power(polar[0], 2))
        ax.set_title(f'Polarisation: {polarisation}')

    else:
        print('Wrong parameters have been passed')


def plot_cnn_history(history):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 8))
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'val'], loc='upper left')

    ax3.plot(history.history['auc'])
    ax3.plot(history.history['val_auc'])
    ax3.set_ylabel('PR-AUC')
    ax3.set_xlabel('epoch')
    ax3.legend(['train', 'val'], loc='upper left')

    ax4.plot(history.history['auc_1'])
    ax4.plot(history.history['val_auc_1'])
    ax4.set_ylabel('ROC-AUC')
    ax4.set_xlabel('epoch')
    ax4.legend(['train', 'val'], loc='upper left')

    return fig


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


def plot_distribution(X, y, features_names, class_map, type='inter'):
    """
    Plots inter or intra class distribution. If `type` is set to 'inter' then only one feature_name is used i.e. if more
    feature names are present in `features_names` then first is selected. If `type` is set to 'intra' then only
    one target name from `class_map` is used i.e. if there is more elements in `class_map` then first one is used.

    Parameters
    ----------
    X: numpy.ndarray
        Data containing features. Shape (n_samples, n_features(names))
    y: numpy.array
        Integer encoded target names.Shape (n_samples, )
    features_names: tuple
        Tuple of strings specifying names of features.
    class_map : dict
        Dictionary where keys are target names and values are theirs integer encoded representation.
        E.g. {'water': 1, 'building': 3}
    type: str
        Specifies type of distribution. Use `inter` for inter class distribution plot and
        `intra` for intra-class distribution plot.
    Returns
    -------
    """
    if type not in ['inter', 'intra']:
        raise Exception(f"Invalid plot type {type}. \n"
                        f"Possible values are ['inter', 'intra']")
    if X.shape[0] != y.shape[0]:
        raise Exception(f"Size of first dimension of samples {X.shape[0]} and targets {y.shape[0]} "
                        f"do not correspond to each other.")
    if y.ndim != 1:
        raise Exception(f"Integer encoded array of targets must be one dimensional.")

    if X.ndim == 1 and len(features_names) != 1:
        raise Exception(f"Number of features in `X` {1 if X.ndim == 1 else X.shape[1]} does not correspond "
                        f"to number of features in `features_names` {len(features_names)}")
    elif X.ndim == 2:
        if X.shape[1] != len(features_names):
            raise Exception(f"Number of features in `X` {1 if X.ndim == 1 else X.shape[1]} does not correspond "
                            f"to number of features in `features_names` {len(features_names)}")

    # CREATE AUXILIARY DATAFRAME
    data = pd.DataFrame(columns=features_names, data=X)

    if type == 'inter':
        map_ = {key: data.loc[np.where(y == val)[0], features_names[0]] for key, val in class_map.items()}
        data = pd.DataFrame(map_)
    else:
        data = data.loc[np.where(y == class_map[[i for i in class_map.keys()][0]])[0], features_names]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title(f"Intra-class distribution of class {[i for i in class_map.keys()][0]}." if type == "intra" else
                 f"Inter class distribution of {features_names[0]}.")
    sns.histplot(data, element='poly', fill=False, ax=ax, bins=50)

    return fig


def intra_pixels_distribution(data_, info_, faultless_map, out_path, path_name='INTRA_pixels_base',
                              labels=('water', 'asphalt', 'grass', 'trees', 'cemetery', 'agri',
                                      'building', 'rock', 'square'), overwrite=False):
    """
    Saves the plots of intra class pixels distributions
    Parameters
    ----------
    data_: list
        First output of `Dataset` class
    info_: list
        Third output of `Dataset` class
    faultless_map: array
        Array of indices of images which are faultless
    path_name: str
        Name of path (one word).
    out_path: str
        Absolute path prior to `path_name`. With `path_name` will be used as `out_path/DISTRIBUTIONS/path_name`
    labels: tuple
        Names of classes
    overwrite: bool
        Whether to overwrite already existing image.
    Returns
    -------

    """
    if not os.path.isdir(os.path.join(out_path, 'DISTRIBUTIONS', path_name)):
        try:
            os.mkdir(os.path.join(out_path, 'DISTRIBUTIONS', path_name))
        except OSError:
            print(f"Creation of the directory {os.path.join(out_path, 'DISTRIBUTIONS', path_name)} failed")

    for name in labels:
        for i, type_ in enumerate(
                [(['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B11', 'B12', 'B8A'], (0, 9), 'BANDS'),
                 (['NDVI', 'NDMI', 'MNDWI', 'NDBI', 'NBRI'], (9, 14), 'INDICES')]):
            # indices of images which belong to particular class
            target = [j for j, inf in enumerate(info_) if inf[0][-1] == name]

            # list of images which are fautless and belong to particular class
            d = [dat for j, dat in enumerate(data_) if j in target and faultless_map[j]]

            # we need to flatten those images within time and bands
            g = [j.reshape(j.shape[0], j.shape[1], -1) for j in d]

            # now concatenate images in first time and particular band
            for_data = [np.concatenate([a[0, k] for a in g])[np.nonzero(np.concatenate([a[0, k] for a in g]))]
                        for k
                        in range(type_[1][0], type_[1][1])]

            c = {k: v for k, v in zip(type_[0], for_data)}
            s = pd.concat([pd.DataFrame(columns=[k], data=v) for k, v in c.items()], ignore_index=False, axis=1)

            if not os.path.isfile(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                               f'intra_class_pixels_{name}_{type_[-1]}.pdf')) or overwrite:
                fig, ax = plt.subplots(figsize=(20, 10))
                ax.set_title(f"Intra {name} class pixels distributions.")

                if i == 0:
                    ax.set_xlim(0, 5000)
                else:
                    ax.set_xlim(0, 10500)
                sns.histplot(s, element='poly', ax=ax, stat='probability', fill=False, bins=100)
                fig.savefig(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                         f'intra_class_pixels_{name}_{type_[-1]}.pdf'))
                plt.clf()
            else:
                print(
                    f"Plot {os.path.join(out_path, 'DISTRIBUTIONS', path_name, f'intra_class_pixels_{name}_{type_[-1]}.pdf')}"
                    f" already exists")


def inter_pixel_distributions(data_, info_, faultless_map, out_path, path_name='INTER_pixels_base',
                              labels=('water', 'asphalt', 'grass', 'trees', 'cemetery', 'agri',
                                      'building', 'rock', 'square'), overwrite=False):
    """
    Saves the plots of inter pixel distributions
    Parameters
    ----------
    data_: list
        First output of `Dataset` class
    info_: list
        Third output of `Dataset` class
    faultless_map: array
        Array of indices of images which are faultless
    path_name: str
        Name of path (one word).
    out_path: str
        Absolute path prior to `path_name`. With `path_name` will be used as `out_path/path_name`
    labels: tuple
        Names of classes
    overwrite: bool
        Whether to overwrite already existing image.
    Returns
    -------

    """
    if not os.path.isdir(os.path.join(out_path, 'DISTRIBUTIONS', path_name)):
        try:
            os.mkdir(os.path.join(out_path, 'DISTRIBUTIONS', path_name))
        except OSError:
            print(f"Creation of the directory {os.path.join(out_path, 'DISTRIBUTIONS', path_name)} failed")

    for i, name in enumerate(
            ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B11', 'B12', 'B8A', 'NDVI', 'NDMI', 'MNDWI', 'NDBI',
             'NBRI']):
        # list of lists of indices of particular class
        targets = [[j for j, inf in enumerate(info_) if inf[0][-1] == label] for label in labels]

        # list of lists of images which are faultless and within particular class
        d = [[dat for j, dat in enumerate(data_) if j in target and faultless_map[j]] for target in targets]

        # we need to flatten those images within time and bands
        g = [[k.reshape(k.shape[0], k.shape[1], -1) for k in j] for j in d]
        h = [np.concatenate([a[0, i] for a in m]) for m in g]
        for_data = [p[np.nonzero(p)] for p in h]

        c = {k: v for k, v in zip(labels, for_data)}
        s = pd.concat([pd.DataFrame(columns=[k], data=v) for k, v in c.items()], ignore_index=False, axis=1)

        if not os.path.isfile(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                           f'inter_class_pixels_{name}.pdf')) or overwrite:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.set_title(f"Inter class pixel distribution within band {name}.")

            sns.histplot(s.iloc[0:50000], element='poly', ax=ax, stat='probability', fill=False, bins=100)
            fig.savefig(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                     f'inter_class_pixels_{name}.pdf'))
            plt.clf()
        else:
            print(
                f"Plot {os.path.join(out_path, 'DISTRIBUTIONS', path_name, f'inter_class_pixels_{name}.pdf')}"
                f" already exists")


def inter_features_distributions(X, y, mapping, out_path, path_name='INTER_features_base',
                                 overwrite=False):
    """
    Generates inter class features (mean, std) distributions and saves plots to particular directory
    Parameters
    ----------
    out_path: str
        Absolute path prior to `path_name`. With `path_name` will be used as `out_path/DISTRIBUTIONS/path_name`
    X: numpy array
        Data(features) to be predicted.
    y: numpy array
        Targets.
    mapping: dict
    path_name: str
        Name of path (one word).
    overwrite: bool
        Whether to overwrite already existing image.

    """
    # ORDER -> [B02, B03, B04, B05, B06, B07, B11, B12, B8A, 'NDVI', 'NDMI', 'MNDWI', 'NDBI', 'NBRI']
    if not os.path.isdir(os.path.join(out_path, 'DISTRIBUTIONS', path_name)):
        try:
            os.mkdir(os.path.join(out_path, 'DISTRIBUTIONS', path_name))
        except OSError:
            print(f"Creation of the directory {os.path.join(out_path, 'DISTRIBUTIONS', path_name)} failed")
    names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B11', 'B12', 'B8A', 'NDVI', 'NDMI', 'MNDWI', 'NDBI', 'NBRI']
    for i, name in enumerate(names):
        for j, name_2 in enumerate([' MEAN', ' STD']):

            if not os.path.isfile(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                               'inter_class_' + name + name_2)) or overwrite:
                plot = plot_distribution(X[:, 2 * i + j], np.argmax(y, axis=1), type='inter',
                                         features_names=[name + name_2],
                                         class_map=mapping)
                plot.savefig(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                          'inter_class_' + name + name_2 + '.pdf'))
            else:
                print(
                    f"Plot {os.path.join(out_path, 'DISTRIBUTIONS', path_name, 'inter_class_' + name + name_2 + '.pdf')}"
                    f" already exists")


def intra_features_distributions(X, y, mapping, out_path, path_name='INTRA_features_base',
                                 overwrite=False):
    """
    Generates intra class features(mean, std) distributions and saves plots to particular directory
    Parameters
    ----------
   out_path: str
        Absolute path prior to `path_name`. With `path_name` will be used as `out_path/DISTRIBUTIONS/path_name`
    X: numpy array
        Data(features) to be predicted.
    y: numpy array
        Targets.
    mapping: dict
    path_name: str
        Name of path (one word).
    overwrite: bool
        Whether to overwrite already existing image.

    """
    # ORDER -> [B02, B03, B04, B05, B06, B07, B11, B12, B8A, 'NDVI', 'NDMI', 'MNDWI', 'NDBI', 'NBRI']
    if not os.path.isdir(os.path.join(out_path, 'DISTRIBUTIONS', path_name)):
        try:
            os.mkdir(os.path.join(out_path, 'DISTRIBUTIONS', path_name))
        except OSError:
            print(f"Creation of the directory {os.path.join(out_path, 'DISTRIBUTIONS', path_name)} failed")

    for i, name in enumerate(mapping.keys()):
        for j, type_ in enumerate([' MEAN', ' STD']):

            if not os.path.isfile(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                               f'intra_class_{name}_BANDS_{type_}.pdf')) or overwrite:

                # MEANS ARE ON EVEN POSITION AND STDS ARE ON ODD POSITION IN X VECTOR
                # FIRSTLY GO ONLY THROUGH BANDS
                plot = plot_distribution(X[:, 0 + j:18:2], np.argmax(y, axis=1), type='intra',
                                         features_names=[a + '_' + type_ for a in ['B02', 'B03', 'B04', 'B05',
                                                                                   'B06', 'B07', 'B11', 'B12',
                                                                                   'B8A']],
                                         class_map={name: i})

                plot.savefig(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                          f'intra_class_{name}_BANDS_{type_}.pdf'))
            else:
                print(
                    f"Plot {os.path.join(out_path, 'DISTRIBUTIONS', path_name, f'intra_class_{name}_BANDS_{type_}.pdf')}"
                    f" already exists")

    for i, name in enumerate(mapping.keys()):
        for j, type_ in enumerate([' MEAN', ' STD']):

            if not os.path.isfile(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                               f'intra_class_{name}_INDICES_{type_}.pdf')) or overwrite:

                # MEANS ARE ON EVEN POSITION AND STDS ARE ON ODD POSITION IN X VECTOR
                # NOW ITERATE THROUGH INDICES
                plot = plot_distribution(X[:, 18 + j:X.size:2], np.argmax(y, axis=1), type='intra',
                                         features_names=[a + '_' + type_ for a in
                                                         ['NDVI', 'NDMI', 'MNDWI', 'NDBI', 'NBRI']],
                                         class_map={name: i})
                plot.savefig(os.path.join(out_path, 'DISTRIBUTIONS', path_name,
                                          f'intra_class_{name}_INDICES_{type_}.pdf'))
            else:
                print(
                    f"Plot {os.path.join(out_path, 'DISTRIBUTIONS', path_name, f'intra_class_{name}_INDICES_{type_}.pdf')}"
                    f" already exists")


def plot_sizes_distribution(data):
    """
    Plots distributions of sizes of images in dataset
    Parameters
    ----------
    data: list
         First output of `Dataset` class method `load`

    """

    widths = [data[0][i].shape[2] for i in range(len(data[0]))]
    heights = [data[0][i].shape[3] for i in range(len(data[0]))]

    w = np.array(widths)
    h = np.array(heights)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title(f'Distribution of widths of images within dataset')
    ax2.set_title(f'Distribution of heights of images within dataset')
    ax1.set_xlabel('Width in pixels.')
    ax2.set_xlabel('Height in pixels.')
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    sns.histplot(w, ax=ax1, binwidth=3)
    sns.histplot(h, ax=ax2, binwidth=3)

    return fig


def plot_correlations(X, y, features_names, class_map):
    """
    Plots correlation matrix (Pearson).
    Parameters
    ----------
    X: numpy.ndarray
        Data containing features.Shape (n_samples, n_features(names))
    y: numpy.ndarray
        One-hot encoded target names. Shape (n_samples, n_targets)
    features_names: tuple
        Tuple of strings specifying names of features.
    class_map : dict
        Dictionary where keys are target names and values are theirs integer encoded representation.
        E.g. {'water': 1, 'building': 3}

    Returns
    -------

    """

    # CREATE AUXILIARY DATAFRAMES
    features = pd.DataFrame(columns=features_names, data=X)
    labels = pd.DataFrame(data=y, columns=[i for i in class_map.keys()])
    final = pd.concat([features, labels], axis=1)

    # CALCULATE PEARSON CORRELATION MATRIX
    c = final.corr(method='pearson')

    mask = np.zeros_like(c)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title("Correlation")
    sns.heatmap(c, annot=True, mask=mask, ax=ax, cmap="YlGnBu")

    return fig


def plot_feature_importances(importances, std, indices, num_features):
    """
    Plots feature importances (MDI).
    Parameters
    ----------
    importances : np.array
    std : np.array
        standard deviation
    indices : np.array
        descending ordered indices of features
    X : np.ndarray
    features_map: np.array
        shape [num_features, 1] of boolean type indicating whether particular feature was used
    Returns
    -------
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title("Feature importances")  # ax.set_title("Dôležitosť prediktora")
    features_names = ('B02', 'B02', 'B03', 'B03', 'B04', 'B04', 'B05', 'B05', 'B06', 'B06', 'B07',
                      'B07', 'B11', 'B11', 'B12', 'B12', 'B8A', 'B8A', 'NDVI', 'NDVI', 'NDMI', 'NDMI',
                      'NDWI', 'NDWI', 'NDBI', 'NDBI', 'NBRI', 'NBRI')
    names = np.array([name + '_mean' if i % 2 == 0 else name + '_std' for i, name in enumerate(features_names)])
    ax.bar(names[indices], importances[indices],
           color="r", yerr=std[indices], align="center")
    # ax.set_xticks(range(num_features))
    # ax.set_xticklabels(indices)
    ax.set_xlim([-1, num_features])

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
    ax.set_title(f'Pravdepodobnostná maska.')

    return fig


def plot_lulc(data, labels, cmap, reclass_rules=None):
    if reclass_rules is not None:
        uniq = np.unique(data)
        cmap = {i: cmap[reclass_rules[i]] for i in range(uniq[-1]+1)}
        labels = [labels[reclass_rules[i]] for i in range(uniq[-1]+1)]
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
    #img = adjust_log(data, 0.06).transpose(1, 2, 0)
    img = data.transpose(1, 2, 0)
    ax.imshow(img / np.quantile(img, 0.99))
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    return fig


# TODO get some inspiration -> https://datavizproject.com/
