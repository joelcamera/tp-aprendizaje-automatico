class ParameterTuning():
    
    def __init__(self, decisionTreeCriterion, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.decisionTreeCriterion = decisionTreeCriterion
        
    def train(self, parameters_to_tune_array, parameter_name):
        def get_dec_tree_class(random_state=42, max_depth=None, min_samples_leaf=1, max_features=None, max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0):
            return DecisionTreeClassifier(criterion=self.decisionTreeCriterion, 
                                        random_state=random_state,
                                        max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features,
                                        max_leaf_nodes=max_leaf_nodes,
                                        min_samples_split=min_samples_split,
                                        class_weight=None,
                                        min_weight_fraction_leaf = min_weight_fraction_leaf,
                                        splitter='best')
            
        print("parameters tuning for {}:{}".format(parameter_name, parameters_to_tune_array))
        train_results = []
        test_results = []
        max_depth=None
        min_samples_split=2
        min_samples_leaf=1
        for curr_parameter in parameters_to_tune_array:
            if parameter_name == 'max_depth':
                max_depth = curr_parameter
            elif parameter_name == 'min_samples_split':
                min_samples_split = curr_parameter
            elif parameter_name == 'min_samples_leaf':
                min_samples_leaf = curr_parameter
                
            dt = get_dec_tree_class(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            dt.fit(self.X_train, self.y_train)
            train_pred = dt.predict(self.X_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            # Add auc score to previous train results
            train_results.append(roc_auc)
            
            y_pred = dt.predict(self.X_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            # Add auc score to previous test results
            test_results.append(roc_auc)
            
        line1, = plt.plot(parameters_to_tune_array, train_results, 'b', label='Train AUC')
        line2, = plt.plot(parameters_to_tune_array, test_results, 'r', label='Test AUC')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel(parameter_name)
        plt.show()
        
        #TODO: Revisar si esto tiene sentido!!
        ret = list(zip(list(map(lambda x, y: 0 if abs(x-y) <= 0.005 else abs(x-y), test_results, train_results)), test_results, parameters_to_tune_array))
        #Ordeno de mayor a menor los resultados de auc para test
        ret.sort(key=lambda x: (-x[1]) )
        #Ordeno de menor a mayor la diferencia de auc entre train y test
        ret.sort(key=lambda x: (x[0]) )
        print("Mejores parámetros para '{}': {}".format(parameter_name,list(map(lambda x: x[2], ret))))
        return ret
    
    def get_best_max_depth(self, hasta):
        inicio = time.time()
        max_depths = np.linspace(1, hasta, int(hasta/2), dtype="int", endpoint=True)
        best_values = self.train(max_depths, 'max_depth')
        fin = time.time()
        print("Tiempo total (min): {}\n".format(round((fin-inicio)/60, 2)))
        return best_values
        
    def get_best_min_samples_split(self, hasta=1.0):
        inicio = time.time()
        if hasta == 1.0:
            min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
        else:
            min_samples_splits = np.linspace(2, hasta, int(hasta/2), dtype="int", endpoint=True)
        best_values = self.train(min_samples_splits, 'min_samples_split')
        fin = time.time()
        print("Tiempo total (min): {}\n".format(round((fin-inicio)/60, 2)))
        return best_values
    
    def get_best_min_samples_leaf(self, hasta=1.0):
        inicio = time.time()
        if hasta == 1.0:
            min_samples_leafs = np.linspace(0.1, 1.0, 10, endpoint=True)
        else:
            min_samples_leafs = np.linspace(1, hasta, int(hasta/2), dtype="int", endpoint=True)
        best_values = self.train(min_samples_leafs, 'min_samples_leaf')
        fin = time.time()
        print("Tiempo total (min): {}\n".format(round((fin-inicio)/60, 2)))
        return best_values
    

    @staticmethod
    def list_only_parameters(l):
        return [t[2] for t in l]

    @staticmethod
    def get_best_result(ret1, ret2):#ret example [(abs(trai-test), auc_test, max_depth)]
        avg1 = reduce(lambda x,y: (x[0]+y[0],x[1]+y[1]), ret1)
        avg1 = avg1[1]/len(ret1)
        avg2 = reduce(lambda x,y: (x[0]+y[0],x[1]+y[1]), ret2)
        avg2 = avg2[1]/len(ret2)
        ret = ret1 if avg1 > avg2 else ret2
        return ParameterTuning.list_only_parameters(ret)