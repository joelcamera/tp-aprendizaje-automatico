from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn import preprocessing
from sklearn import metrics

import time
import matplotlib.pyplot as plt

class AbstractClassificationProblem:
    labels = ['No Fraude', 'Fraude']
    desc = ""
    
    def train(self):
        print("Comienza Entrenamiento")
        inicio = time.time()
        self.print_self()
        self.clf_model.fit(self.X_train, self.y_train)
        self.y_predict = self.clf_model.predict(self.X_test)
        print("Entrenado")       
        fin = time.time()
        print("Tiempo total (min): {}".format(round((fin-inicio)/60, 2)))
    
    def show_confusion_matrix(self):
        self.print_self()
        ConfusionMatrixDisplay.from_estimator(estimator=self.clf_model,
                                              X=self.X_test, 
                                              y=self.y_test,
                                              display_labels=self.labels)
        plt.show()
    
    def show_classification_report(self):
        self.print_self()
        print(classification_report(self.y_test, self.y_predict, target_names=self.labels))

    def accuracy(self):
        return accuracy_score(self.y_test, self.y_predict)
    
    def recall(self):
        return recall_score(self.y_test, self.y_predict)
    
    def precision_score(self):
        return precision_score(self.y_test, self.y_predict)
    
    def f1_score(self):
        return f1_score(self.y_test, self.y_predict)
    
    def print_self(self):
        pass
    
    def predict_proba(self, x):
        return self.clf_model.predict_proba(x)
    
    def auc(self):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test, self.y_predict)
        return auc(false_positive_rate, true_positive_rate)
    
    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_predict, pos_label=1)
        auc = metrics.roc_auc_score(self.y_test, self.y_predict)
        fig, ax = plt.subplots()
        ax.plot(fpr,tpr,label="["+self.tipo+"] - AUC="+str(auc))
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
        plt.title(f'Curva ROC')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend() 

class AbstractDecisionTree(AbstractClassificationProblem):
    criterion = ""
    tipo = ""
    
    def __init__(self, X_train, X_test, y_train, y_test, target=[0,1], max_depth=None, min_samples_leaf=1, min_samples_split=2, desc=""):
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.clf_model = DecisionTreeClassifier(criterion=self.criterion, 
                                                random_state=42,
                                                max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf,
                                                min_samples_split=min_samples_split,
                                                max_features=None,
                                                max_leaf_nodes=None,
                                                class_weight=None,
                                                splitter='best')
        self.desc = desc

        self.target = target
        self.feature_names = list(X_train.columns)
        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
    
    def show_matrix(self):
        dot_data = tree.export_graphviz(self.clf_model,
                                        out_file=None,
                                        feature_names=self.feature_names,
                                        class_names=str(self.target),
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        dot_data = StringIO()
        export_graphviz(self.clf_model, 
                        out_file=dot_data, 
                        filled=True, 
                        rounded=True, 
                        special_characters=True,
                        feature_names=self.feature_names,
                        class_names=self.labels)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        return Image(graph.create_png())
    
    def print_self(self):
        print("**", "Arbol de decision - " + self.tipo + " - " + self.desc, "**")
        print('**', "max_depth=" + str(self.max_depth) + ",", "min_samples_leaf=" + str(self.min_samples_leaf), "min_samples_split=" + str(self.min_samples_split), "**")


class GiniDecisionTree(AbstractDecisionTree):
    criterion = "gini"
    tipo = "Gini Index"

class InformationGainDecisionTree(AbstractDecisionTree):
    criterion = "entropy"
    tipo = "Information Gain"

# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
class AbstractNeuralNetwork(AbstractClassificationProblem):
    solver = ''
    tipo = ''
    
    def __init__(self, X, y, X_train, X_test, y_train, y_test, alpha=1e-5, hidden_layer_sizes=(15,), max_iter=5000, desc=""):
        self.X = X
        self.y = y
        
        # https://scikit-learn.org/stable/modules/preprocessing.html
        self.X_train = self._scale(X_train)
        self.X_test = self._scale(X_test)
        
        #https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
        self.y_train = y_train.values.ravel()

        self.y_test = y_test
        
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        
        self.desc=desc

        self.clf_model = MLPClassifier(solver=self.solver,
                                       alpha=alpha, 
                                       hidden_layer_sizes=hidden_layer_sizes, 
                                       random_state=42, 
                                       max_iter=max_iter)   
    
    def print_self(self):
        print("**", "Red Neuronal - " + self.tipo, "**")
        print("**", 
              "alpha=" + str(self.alpha), 
              "hidden_layer_sizes=" + str(self.hidden_layer_sizes), 
              "max_iter=" + str(self.max_iter), 
              "**")

    def _scale(self, X):
        scaler = preprocessing.StandardScaler().fit(X)
        return scaler.transform(X)


class LBFGSNeuralNetwork(AbstractNeuralNetwork):
    solver = 'lbfgs'
    tipo = "LBFGS"

class SGDNeuralNetwork(AbstractNeuralNetwork):
    solver = 'sgd'
    tipo = "SGD"

class AdamNeuralNetwork(AbstractNeuralNetwork):
    solver = 'adam'
    tipo = 'ADAM'