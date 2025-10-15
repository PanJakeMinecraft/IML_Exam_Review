import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.multiclass import unique_labels

class CarModel:
    def __init__(self):
        self.encoder = OrdinalEncoder()
        self.model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)

    def CreateDataset(self):
        # FOR THE DATASET, I find it from ChatGPT
        data = {
            'buying': ['low', 'low', 'med', 'med', 'high', 'high', 'vhigh', 'vhigh'] * 3,
            'maint': ['low', 'med', 'high', 'vhigh', 'low', 'med', 'high', 'vhigh'] * 3,
            'safety': ['low', 'med', 'high'] * 8,
            'class': ['unacc', 'acc', 'good', 'unacc', 'vgood', 'unacc', 'acc', 'good'] * 3
        }

        self.features = ["buying", "maint", "safety"]
        self.x = []

        for i in range(len(data["class"])):
            row = []
            for f in self.features:
                row.append(data[f][i])
            self.x.append(row)
        
        self.x = np.array(self.x)
        self.y = np.array(data["class"])

    def PreProcess(self):
        self.x_encod = self.encoder.fit_transform(self.x)

    def train(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_encod, self.y, test_size=0.3, random_state=0 
        )
        self.model.fit(self.x_train, self.y_train)
    
    def Evaluate(self):
        self.y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        return accuracy
    
    def Predict(self, input_values):
        encod_input = self.encoder.transform([input_values])
        predict = self.model.predict(encod_input)
        prob = self.model.predict_proba(encod_input)
        print("Prediction for {0} = {1}".format(input_values, predict[0]))
        print("Class Prediction: {0}".format(dict(zip(self.model.classes_, prob[0]))))
        return predict[0], prob[0]

    def Vizualize(self, title):
        plt.figure(figsize=(16, 8))
        plot_tree(self.model, 
                  feature_names=self.features, 
                  class_names=self.model.classes_,
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title(title, fontsize = 16)
        plt.show()

if __name__ == "__main__":
    model = CarModel()
    model.CreateDataset()
    model.PreProcess()
    model.train()
    print(model.Evaluate())
    model.Predict(["low", "med", "high"])
    model.Predict(["vhigh", "vhigh", "med"])
    model.Vizualize(title="Car Maintainance Tree")