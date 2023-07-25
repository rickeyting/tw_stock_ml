from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_acc_roc(test_output, test_y):
    thresholds = np.arange(-0.1, 0.1, 0.001)
    aucs = []
    for threshold in thresholds:
        binary_predictions = (test_output > threshold).numpy()
        binary_test_y = (test_y > 0).numpy()  # transform test_y to binary form
        fpr, tpr, _ = roc_curve(binary_test_y, binary_predictions)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Now plot the ROC curve
    plt.figure()
    plt.plot(thresholds, aucs)
    plt.xlabel('Threshold')
    plt.ylabel('Area Under ROC')
    plt.title('Threshold versus Area Under ROC')
    plt.show()


def plot_roc(test_output, test_y):
    thresholds = np.arange(-0.1, 0.1, 0.001)
    test_output_np = test_output.numpy()
    test_y_np = (test_y > 0).numpy()

    tprs = []
    fprs = []
    for threshold in thresholds:
        binary_predictions = (test_output_np > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_y_np, binary_predictions).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tprs.append(tpr)
        fprs.append(fpr)
    # Now plot the ROC curve
    plt.figure()
    plt.plot(fprs, tprs)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.show()



