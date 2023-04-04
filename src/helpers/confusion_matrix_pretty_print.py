# -*- coding: utf-8 -*-
"""
plot a pretty confusion matrix with seaborn
Created on Mon Jun 25 14:17:37 2018
@author: Wagner Cipriano - wagnerbhbr - gmail - CEFETMG / MMC
REFerences:
  https://www.mathworks.com/help/nnet/ref/plotconfusion.html
  https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
  https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
  https://www.programcreek.com/python/example/96197/seaborn.heatmap
  https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/31720054
  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
"""

# imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker
from matplotlib.collections import QuadMesh
import seaborn as sn


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0,
                               normalize=False, scientific=False):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    shape = array_df.shape
    text_add = [];
    text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-3][-3]
    per = (float(cell_val) / tot_all) * 100  # percent per cell
    curr_column = array_df[:, col]
    ccl = len(curr_column)
    per_ok = 0

    if shape[0] == shape[1]:
        # last three lines  or  columns
        if (col >= shape[1] - 3) or (lin >= shape[0] - 3):
            # tots and percents
            if (cell_val != 0):
                if (col == ccl - 3) or (lin == ccl - 3):
                    per_ok = (array_df[lin, col] / array_df[-3, -3]) * 100
                else:
                    per_ok = array_df[lin, col] * 100

            else:
                per_ok = per_err = 0

            per_ok_s = ['%.2f%%' % (per_ok), '100%'][int(per_ok == 100)]

            # text to DEL
            text_del.append(oText)

            # text to ADD
            font_prop = fm.FontProperties(weight='bold', size=fz-1)
            text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
            if [lin, col] in [[shape[1] - 3, shape[1] - 2], [shape[1] - 3, shape[1] - 1], [shape[1] - 2, shape[1] - 3],
                              [shape[1] - 2, shape[1] - 3], [shape[1] - 2, shape[1] - 1], [shape[1] - 1, shape[1] - 3],
                              [shape[1] - 1, shape[1] - 2]]:
                lis_txt = ['']
            elif (col >= shape[1] - 2) or (lin >= shape[0] - 2):
                lis_txt = [per_ok_s]
            else:
                if scientific:
                    lis_txt = ["{:.2e}".format(cell_val), per_ok_s]
                else:
                    lis_txt = ['%d' % (cell_val), per_ok_s]
            lis_kwa = [text_kwargs]
            dic = text_kwargs.copy();
            dic['color'] = 'y';
            lis_kwa.append(dic);
            # dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
            lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
            for i in range(len(lis_txt)):
                newText = dict(x=lis_pos[i][0],
                               y=lis_pos[i + 1 if len(lis_txt) == 1 else i][1],
                               text=lis_txt[i], kw=lis_kwa[i + 1 if len(lis_txt) == 1 else i])
                # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
                text_add.append(newText)
            # print '\n'

            # set background color for sum cells (last line and last column)
            carr = [0.27, 0.30, 0.27, 1.0]  # grey
            if (col >= shape[1] - 3) and (lin >= shape[0] - 3):
                carr = [0.17, 0.20, 0.17, 1.0]  # black
            facecolors[posi] = carr

        else:
            if (per > 0) and not normalize:
                if scientific:
                    txt = f'{"{:.2e}".format(cell_val)}\n{"%.2f%%" % per}'
                else:
                    txt = '%d\n%.2f%%' % (cell_val, per)
            elif (per > 0):
                txt = '%.2f%%' % (per)
            else:
                if (show_null_values == 0):
                    txt = ''
                elif (show_null_values == 1):
                    txt = '0'
                else:
                    txt = '0\n0.0%'
            oText.set_text(txt)

            # main diagonal
            if (col == lin):
                # set color of the textin the diagonal to white
                oText.set_color('w')
                # set background color in the diagonal to blue
                facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
            else:
                oText.set_color('k')

        return text_add, text_del


#

def insert_totals(df_cm, print_IoU=True, ignore_index=[]):
    """ insert total column and line (the last ones) """

    relevant = [i for i in range(df_cm.shape[0]) if i not in ignore_index]

    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())

    if ignore_index:
        sum_col_correct = []
        for c in df_cm.columns:
            sum_col_correct.append(df_cm[c].iloc[relevant].sum())
    else:
        sum_col_correct = sum_col

    diagonal = [df_cm.iloc[i, i] for i in range(len(sum_col))]
    if ignore_index:
        # correct
        trace = np.array(diagonal)[relevant].sum()
    else:
        trace = np.array(diagonal).sum()
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())

    if ignore_index:
        sum_lin_correct = []
        for item_line in df_cm.iterrows():
            sum_lin_correct.append(item_line[1].iloc[relevant].sum())
    else:
        sum_lin_correct = sum_lin
    df_cm['Sum(actual)'] = sum_lin #df_cm['Suma(skutoč)'] = sum_lin

    sum_col.append(np.sum(sum_lin))
    df_cm.loc['Sum(predic)'] = sum_col #df_cm.loc['Suma(pred)'] = sum_col

    if ignore_index:
        sum_col_correct.append(np.sum(np.array(sum_lin_correct)[relevant]))
    else:
        sum_col_correct = sum_col

    # recall
    recall = [diag / actual if actual != 0 else 0 for actual, diag in zip(sum_lin_correct, diagonal)]
    for i in ignore_index:
        recall[i] = -0.0
    recall.append(0)
    df_cm['Recall'] = recall

    # precision
    precision = [diag / predicted if predicted != 0 else 0 for predicted, diag in zip(sum_col_correct[:-1], diagonal)]
    for i in ignore_index:
        precision[i] = -0.0
    precision.extend([0, trace / sum_col_correct[-1]])
    df_cm.loc['Precision'] = precision


    if print_IoU:
        # intersection over union
        IoU = [diag / (total + predicted - diag) if total + predicted - diag != 0 else 0
               for diag, total, predicted in zip(diagonal, sum_lin_correct, sum_col_correct[:-1])]
        for i in ignore_index:
            IoU[i] = -0.0

        # Jaccard aka mean IoU
        jaccard = np.array(IoU)[relevant].mean()
        IoU.extend([0, 0])
        df_cm['IoU'] = IoU

        # f1 score aka dice coefficient
        f1_score_aka_dice = [2 * diag / (total + predicted) if total + predicted != 0 else 0
                             for diag, total, predicted in zip(diagonal, sum_lin_correct, sum_col_correct[:-1])]
        for i in ignore_index:
            f1_score_aka_dice[i] = -0.0
        f1_score_aka_dice.extend([0, 0, jaccard])
        df_cm.loc['F1 score'] = f1_score_aka_dice
    else:
        df_cm[''] = [0 for i in range(len(sum_lin) + 2)]

        # f1 score aka dice coefficient
        f1_score_aka_dice = [2 * diag / (total + predicted) if total + predicted != 0 else 0
                             for diag, total, predicted in
                             zip(diagonal, sum_lin_correct, sum_col_correct[:-1])]
        for i in ignore_index:
            f1_score_aka_dice[i] = -0.0
        mean_f1 = np.array(f1_score_aka_dice)[relevant].mean()
        f1_score_aka_dice.extend([0, 0, mean_f1])
        df_cm.loc['F1 score'] = f1_score_aka_dice
    # print ('\ndf_cm:\n', df_cm, '\n\b\n')


#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="YlGnBu", fmt='.2f', fz=12,
                                 lw=0.8, cbar=False, figsize=[25, 25], show_null_values=0, pred_val_axis='y',
                                 print_IoU=False,scientific_notation=False,
                                 title='Confusion matrix', normalize=False, ignore_index=[]):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)

    """

    # create "Total" column
    insert_totals(df_cm, print_IoU=print_IoU, ignore_index=ignore_index)

    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted' #'Predikované'
        ylbl = 'Actual' #'Skutočné'
    else:
        xlbl = 'Actual' #'Skutočné'
        ylbl = 'Predicted' #'Predikované'
        df_cm = df_cm.T

    # this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    df_cm.fillna(0.0, inplace=True)
    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"fontsize": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    ax_new1 = ax.twinx()
    ax_new2 = ax_new1.twiny()

    labels = [text.get_text() for text in ax.get_xticklabels()]

    labels[-3] =  'Total' #'Celkovo' #'Total samples'
    labels[-2] = 'Accuracy'
    if print_IoU:
        labels[-1] = 'mIoU'
    else:
        labels[-1] = 'mean F1'

    ticks = [tick for tick in ax.get_xticks()]

    ax_new2.set_xticks(ticks)
    ax_new1.set_yticks(ticks)

    ax_new2.set_xticklabels([text.get_text() for text in ax.get_xticklabels()], fontsize=fz + 5, rotation=-30)  # top
    ax_new1.set_yticklabels(labels, fontsize=fz + 5, rotation=-45)  # right

    # set ticklabels
    ax.set_xticklabels(labels, rotation=30, fontsize=fz + 5)  # bottom
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=fz + 5)  # left

    ax_new2.set_xlim(ax.get_xlim())
    ax_new1.set_ylim(ax.get_ylim())

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = [];
    text_del = [];
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1]);
        col = int(pos[0]);
        posi += 1
        # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values,
                                             normalize=normalize, scientific=scientific_notation)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    #ax.set_title(title, fontsize=fz + 15)
    ax.set_xlabel(xlbl, fontsize=fz + 8)
    ax.set_ylabel(ylbl, fontsize=fz + 8)
    plt.tight_layout()  # set layout slim
    # plt.show()
    return fig


#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
                                    fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0,
                                    pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    # data
    if (not columns):
        # labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        # labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize = [9, 9];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values,
                                 pred_val_axis=pred_val_axis)


#


#
# TEST functions
#
def _test_cm():
    # test function with confusion matrix done
    array = np.random.randint(100000, 100000000, (15, 15))
    # get pandas dataframe
    df_cm = DataFrame(array, index=range(0, 15), columns=range(0, 15))
    # colormap: see this and choose your more dear
    cmap = 'PuRd'
    return pretty_plot_confusion_matrix(df_cm, cmap=cmap, pred_val_axis='x', print_IoU=True,
                                        title='Land Cover confusion matrix', normalize=False, fz=12.5,
                                        ignore_index=[0, 14], scientific_notation=True)


#

def _test_data_class():
    """ test function with y_test (actual values) and predictions (predic) """
    # data
    y_test = np.array(
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
         3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4,
         5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    predic = np.array(
        [1, 2, 4, 3, 5, 1, 2, 4, 3, 5, 1, 2, 3, 4, 4, 1, 4, 3, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2,
         4, 4, 5, 1, 2, 3, 3, 5, 1, 2, 3, 3, 5, 1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 4, 4,
         5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    """
      Examples to validate output (confusion matrix plot)
        actual: 5 and prediction 1   >>  3
        actual: 2 and prediction 4   >>  1
        actual: 3 and prediction 4   >>  10
    """
    columns = []
    annot = True;
    cmap = 'Oranges';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    # size::
    fz = 12;
    figsize = [9, 9];
    if (len(y_test) > 10):
        fz = 9;
        figsize = [14, 14];
    plot_confusion_matrix_from_data(y_test, predic, columns,
                                    annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


#


#
# MAIN function
#
if (__name__ == '__main__'):
    print('__main__')
    print('_test_cm: test function with confusion matrix done\nand pause')

    _test_cm()
    plt.show()
