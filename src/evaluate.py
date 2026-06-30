import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model


def load_trained_model(model_path):
    return load_model(model_path)


def evaluate_model(model, test_data, class_names=None):
    y_prob = model.predict(test_data)
    y_true = np.array(test_data.classes)
    y_pred = np.argmax(y_prob, axis=1)

    if class_names is None:
        class_names = list(test_data.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred)

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'class_names': class_names,
        'classification_report': report,
        'confusion_matrix': confusion,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
    }
