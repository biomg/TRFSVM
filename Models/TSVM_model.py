import numpy as np
from scipy.stats import  pearsonr
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, accuracy_score
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from re import search
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

Train = np.load('MPRA_dataset.npy')
train_label = np.load('MPRA_dataset_label.npy')
Train_label = to_categorical(train_label)


def reset_tlmodel_seq(basemodel):
    iflatten = 0
    for i in range(len(basemodel.layers)):
        if search('flatten', basemodel.layers[i].name) is not None:
            iflatten = i
            break
    riflatten = len(basemodel.layers) - iflatten
    x = basemodel.layers[-riflatten].output
    tlmodel = Model(basemodel.inputs, x)
    for layer in tlmodel.layers[:(iflatten + 1)]:
        layer.trainable = False
    for i, layer in enumerate(tlmodel.layers):
        print(i, layer.name, layer.trainable)
    return tlmodel


def eval_model(y_prob,y_test):
    from sklearn.metrics import matthews_corrcoef
    y_test_prob = y_prob
    y_test_classes = np.argmax(y_test_prob, axis=-1)
    fpr, tpr, thresholds = roc_curve(y_test[:, 0], y_test_prob[:, 0])
    auc_test = auc(fpr, tpr)
    acc_test = accuracy_score(y_test_classes, np.argmax(y_test, axis=-1))
    f1_test = f1_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    recall_test = recall_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    precision_test = precision_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    R_test = pearsonr(y_test[:, 0], y_test_prob[:, 0])[0]
    MCC_test = matthews_corrcoef(np.argmax(y_test, axis=-1), y_test_classes)
    acc_test = round(acc_test, 3)
    auc_test = round(auc_test, 3)
    f1_test = round(f1_test, 3)
    precision_test = round(precision_test, 3)
    recall_test = round(recall_test, 3)
    R_test = round(R_test, 3)
    MCC_test = round(MCC_test, 3)
    return [acc_test, auc_test, f1_test, precision_test, recall_test, R_test,MCC_test]


def get_feature(model, X, imp_result):
    XX = model.predict(X)
    X_feature = XX[:,imp_result]
    return X_feature


def ForestClassifier(x_train, y_train):
    forest = RandomForestClassifier()
    forest.fit(x_train, y_train)
    importance = forest.feature_importances_
    imp_result = np.argsort(importance)[::-1]
    imp_results = imp_result[importance[imp_result] > 0]
    return imp_results


def Predict(model, x_train, y_train,imp_result):
    sv = SVC(C=1, kernel='rbf', probability=True)
    x_feature = get_feature(model, x_train, imp_result)
    sv.fit(x_feature, y_train)
    return sv


def vote_pra(mechain, x_test):
    probs = mechain.predict_proba(x_test)
    return probs


shape = (1, 5)
test_auc = np.zeros(shape)
test_F1 = np.zeros(shape)
test_acc = np.zeros(shape)
test_mcc = np.zeros(shape)
model = load_model(r'E:\pythonProject\pythonProject\hgmd10.seq.h5')
tlmodel = reset_tlmodel_seq(model)

if __name__ == '__main__':
    for i in range(5):
        frac_train = 1
        np.random.seed(i)
        x_train, x_test, y_train, y_test = train_test_split(Train, train_label, stratify=train_label,
                                                            test_size=0.2, random_state=i)
        if frac_train != 1:
            x_train, _, y_train, _ = train_test_split(x_train, y_train, stratify=y_train, test_size=1-frac_train,
                                                                random_state=i)
        if frac_train == 1:
            x_train, _, y_train, _ = train_test_split(Train, train_label, stratify=train_label, test_size=0.2,
                                                                random_state=i)
        y_test = to_categorical(y_test)
        XX = tlmodel.predict(x_train)
        imp_results = ForestClassifier(XX, y_train)
        x_test_feature = get_feature(tlmodel, x_test, imp_results)
        mechain = Predict(tlmodel,x_train,y_train,imp_results)
        y_pro = vote_pra(mechain,x_test_feature)
        test = eval_model(y_pro,y_test)
        test_auc[0, i] = test[1]
        test_F1[0, i] = test[2]
        test_mcc[0, i] = test[6]
        print('test__',test)
    test_SVM={'test_SVM_auc':test_auc,'test_SVM_F1':test_F1}
    print("auc_test = %.3f test_SVM_F1 = %.3f test_mcc=%.3f" % (
    np.mean(test_auc, axis=1), np.mean(test_F1, axis=1), np.mean(test_mcc, axis=1)))
