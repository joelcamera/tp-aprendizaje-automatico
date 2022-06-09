class Comparator:
    models = []
    
    def __init__(self, models):
        self.models = models
    
    def show_confusion_matrix(self):
        for model in self.models:
            model.show_confusion_matrix()
            print("\n")

    def show_classification_report(self):
        for model in self.models:
            model.show_classification_report()
            print("\n")

    def accuracy(self):
        for model in self.models:
            print("accuracy[{}]: {}".format(model.tipo, model.accuracy()))
    
    def recall(self):
        for model in self.models:
            print("recall[{}]: {}".format(model.tipo, model.recall()))
    
    def precision_score(self):
        for model in self.models:
            print("precision_score[{}]: {}".format(model.tipo, model.precision_score()))
            
    def f1_score(self):
        for model in self.models:
            print("f1_score[{}-{}]: {}".format(model.tipo, model.desc, model.f1_score()))
    
    def auc(self):
        for model in self.models:
            print("auc[{}-{}]: {}".format(model.tipo, model.desc,model.auc()))
            
    def roc_curve(self):
        fig, ax = plt.subplots()
        for model in self.models:
            fpr, tpr, thresholds = roc_curve(model.y_test, model.y_predict, pos_label=1)
            auc = metrics.roc_auc_score(model.y_test, model.y_predict)
            #ax.plot(fpr, tpr)
            ax.plot(fpr,tpr,label="["+model.tipo+ "-" + model.desc+"] - AUC="+str(auc))
            plt.title(f'Curva ROC')
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
        ax.legend()