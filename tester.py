import argparse
import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2
import os
from FDA import FDA

parser = argparse.ArgumentParser(description="Flexible Discriminant Analysis")
parser.add_argument("--method", default=None, type=str, help="type of ambiguity set")
parser.add_argument("--rho", default=-1, nargs="+", type=float, help="radius of ambiguity set")
parser.add_argument("--cv", default=1, type=int, help="number of folds for validation")
parser.add_argument("--repeat", default=1, type=int, help="number of test/train split")
parser.add_argument("--dataset", default=None, type=str, help="datafile address")
parser.add_argument("--decision_boundary",default = False, type=bool,help="generate decision boundary plots true or false")
args = parser.parse_args()

DIR_SAVE = os.path.join(os.environ["HOME"], "Dropbox/moment_base_classfication/moment_DRO_results_full3")
if not os.path.exists(DIR_SAVE):
    os.makedirs(DIR_SAVE)
acc_threshold1 = 0
acc_threshold2 = 0
def find_max_score(clf,y_true,y_prob,acc_threshold):
    max_acc = 0
    if acc_threshold == 0:
       for acc_num in y_prob[:,0]:
           cur_score =  accuracy_score(y_true, [clf.labels_[0 if item[0]>acc_num else 1] for item in y_prob])
           if cur_score > max_acc:
               max_acc = cur_score
               acc_threshold = acc_num
    else:
        max_acc = accuracy_score(y_true, [clf.labels_[0 if item[0]>acc_threshold else 1] for item in y_prob])
    return max_acc,acc_threshold
   
def compute_scores(clf, y_true, y_prob):
    global acc_threshold1,acc_threshold2
    
    y_prob_1 = y_prob[:, :clf.n_class_]
    y_prob_2 = y_prob[:, clf.n_class_:]
    fpr1, tpr1, thresholds1 = roc_curve(y_true,  y_prob_1[:, 1], pos_label=1)
    fpr2, tpr2, thresholds2 = roc_curve(y_true,  y_prob_2[:, 1], pos_label=1)

    max_acc1,acc_threshold1 = find_max_score(clf,y_true,y_prob_1,acc_threshold1)
    max_acc2,acc_threshold2 = find_max_score(clf,y_true,y_prob_2,acc_threshold2)
       
    return np.array([max_acc1,max_acc2,
                     accuracy_score(y_true, clf.labels_[y_prob_1.argmax(1)]),
                     accuracy_score(y_true, clf.labels_[y_prob_2.argmax(1)]),
                     auc(fpr1, tpr1),
                     auc(fpr2, tpr2)])

def val_test_score(clf, X_tr, y_tr, X_te, y_te):
    skf = StratifiedKFold(n_splits=args.cv)
    skf.get_n_splits(X_tr, y_tr)
    scores_val = []
    global acc_threshold1,acc_threshold2
    for train_index, val_index in skf.split(X_tr, y_tr):   
        acc_threshold1 = 0
        acc_threshold2 = 0
        X_train, X_val = X_tr[train_index], X_tr[val_index]
        y_train, y_val = y_tr[train_index], y_tr[val_index]
        clf.fit(X_train, y_train)
        compute_scores(clf, y_train, clf.predict_proba(X_train))    # compute threshold
        y_val_prob = clf.predict_proba(X_val)
        scores_val.append(compute_scores(clf, y_val, y_val_prob))

    clf.fit(X_tr, y_tr)
    acc_threshold1 = 0
    acc_threshold2 = 0
    compute_scores(clf, y_tr, clf.predict_proba(X_tr))    # compute threshold
    y_prob = clf.predict_proba(X_te)
    scores_te = compute_scores(clf, y_te, y_prob)
    return np.append(np.mean(scores_val, axis=0), scores_te)

def main():
    f_name = args.dataset[11:-4] #./dataset .... .txt
    DIR_SAVE_DATA = os.path.join(DIR_SAVE,f_name)
    if not os.path.exists(DIR_SAVE_DATA):
        os.makedirs(DIR_SAVE_DATA)
    print (f_name)
    f_name = "";
    if args.method is not None:
        f_name +=  args.method
    if hasattr(args.rho, '__iter__'):
        for rho in args.rho:
            f_name += "_" + str(rho)
    f_name = os.path.join(DIR_SAVE_DATA, f_name + ".csv")
    print("training & testing the {} dataset with {} method and rho {}".format(
        args.dataset[11:-4], args.method, args.rho))
    if os.path.exists(f_name):
        print("the model is already trained")
    else:
        clf = FDA(rule="fda", method=args.method, rho=args.rho)
        data = load_svmlight_file(args.dataset)
        X_data = data[0]
        y_data = data[1]
        labels = np.unique(y_data)
        y_data[y_data == labels[0]] = 0
        y_data[y_data == labels[1]] = 1
        scores = []
        if args.decision_boundary:
            global acc_threshold1
            X_data = X_data.toarray()
            X_train = X_data[0:2000,]
            y_train = y_data[0:2000,]
            X_test = X_data[2000:,]
            y_test = y_data[2000:,]
            print(np.shape(X_train),np.shape(X_test))
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)
            prob1 = clf.predict_proba(X_train)
   
            max_acc1,acc_threshold1 = find_max_score(clf,y_train,prob1[:, :clf.n_class_],acc_threshold1)
            print(acc_threshold1)
            np.savetxt(f_name, np.concatenate((X_test,prob), axis=1), fmt="%0.4f", delimiter=",")

            
        
        else:
            for i in range(args.repeat):

                print("running iteration ", i + 1)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_data.toarray(), y_data, test_size=0.25, random_state=1000+i)
                scores.append(val_test_score(clf, X_train, y_train, X_test, y_test))
            np.savetxt(f_name, 100 * np.array(scores), fmt="%0.2f", delimiter=",")
            print(np.mean(np.array(scores),axis = 0))
            print(scores)

if __name__ == "__main__":
    main()
