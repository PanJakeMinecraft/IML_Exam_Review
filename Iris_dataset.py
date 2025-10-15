import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from matplotlib import pyplot as plt
import scipy as sp

def generate_train(data=None):
    if data is None:
        data = load_iris()
    x = data.data 
    y = data.target
    return train_test_split(x, y, test_size=0.25, random_state=42)

def train_model(x_train, y_train, model=None):
    if model is None:
        model = GaussianNB()
    model.fit(x_train, y_train)
    return model

def predict_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    ##### the confusion matrix
    con_mat = confusion_matrix(y_test, y_pred)
    disp_mat = ConfusionMatrixDisplay(confusion_matrix=con_mat)
    disp_mat.plot(cmap="plasma")
    plt.show()
    return acc, mse

def predict_flower(model, data, features):
    pred = model.predict([features])[0]
    flower_name = data.target_names[pred]
    return flower_name

def create_graph(model, x, y, iris):
    features_names = iris.feature_names
    num_features = len(features_names)
    num_classes = len(np.unique(y))
    x_np = np.array(x)

    fig, axs = plt.subplots(2,2, figsize=(12, 8))
    axs = axs.ravel()

    for idx in range(num_features):
        x_vals = np.linspace(x_np[:, idx].min(), x_np[:, idx].max(), 200)
        for c in range(num_classes):
            mean = model.theta_[c, idx]
            std = np.sqrt(model.var_[c, idx])
            y_vals = sp.stats.norm.pdf(x_vals, mean, std)
            axs[idx].plot(x_vals, y_vals, label = f"Class {c} ({iris.target_names[c]})")
        
        axs[idx].set_title(f"{features_names[idx]} Graph")
        axs[idx].set_xlabel("Feature values")
        axs[idx].set_ylabel("Density of the prob")
        axs[idx].legend()

    plt.suptitle("Iris Features distribution")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": ###### Load all the datas for training
    data = load_iris()
    x_train, x_test, y_train, y_test = generate_train(data)

    gnb = GaussianNB()
    model = train_model(x_train=x_train, y_train=y_train, model=gnb)
    
    acc, mse = predict_model(model, x_test, y_test)
    print(f"MSE = {mse:.4f}")
    print(f"Accuray model = {acc:.4f}")

    ### [sepal_lengtyhm sepal_width, petal_length, petal_width]
    flower_dataset = [5.1, 3.5, 1.4, 0.2]
    print(f"The predicted flower = {predict_flower(model, data, flower_dataset)}")
    
    create_graph(gnb, x_train, y_train, data)