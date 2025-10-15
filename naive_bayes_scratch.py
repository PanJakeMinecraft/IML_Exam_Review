import numpy as np
from sklearn.preprocessing import LabelEncoder

def log_gaussian(x, mu, sigma):
    return np.log(1. / (np.sqrt(2. * np.pi) * sigma)) + (-np.power((x - mu) / sigma, 2.) / 2)

class NaiveBayes:
    def __call__(self, x, y, is_continuous=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.targets, counts = np.unique(self.y, return_counts=True)
        self.Nv = len(self.targets)
        self.Fv = counts.astype(float)
        self.Pv = self.Fv / len(self.y)
        
        if is_continuous is None:
            self.is_continuous = [False] * x.shape[1]
        else:
            self.is_continuous = is_continuous
        
        self.encoders = []
        self.ar = []
        for i in range(x.shape[1]):
            if self.is_continuous[i]:
                self.encoders.append(None)
                self.ar.append([])
            else:
                encoder = LabelEncoder()
                self.x[:, i] = encoder.fit_transform(self.x[:, i])
                self.encoders.append(encoder)
                self.ar.append(encoder.classes_)
        
        self.train()
    
    def train(self):
        m = self.x.shape[1]
        self.Pav = {}
        for j, t in enumerate(self.targets):
            idx = self.y == t
            n = np.sum(idx)
            Pavt = []
            
            for i in range(self.x.shape[1]):
                if self.is_continuous[i]:
                    temp = np.vectorize(float)(self.x[idx, i])
                    mu = np.mean(temp)
                    sigma = np.std(temp)
                    Pavt.append((mu, sigma))
                else:
                    p = {}
                    for ar in range(len(self.ar[i])): 
                        p[ar] = (np.sum(self.x[idx, i] == ar) + 1) / (n + len(self.ar[i]))
                    Pavt.append(p)
            self.Pav[t] = Pavt
    
    def test(self, x):
        x_encoded = np.copy(x)
        for i in range(x.shape[1]):
            if not self.is_continuous[i] and self.encoders[i] is not None:
                for j in range(x.shape[0]):
                    val = x[j, i]
                    if val in self.encoders[i].classes_:
                        x_encoded[j, i] = self.encoders[i].transform([val])[0]
                    else:
                        x_encoded[j, i] = -1 
        
        Z = []
        for sample in x_encoded:
            p = np.zeros(len(self.targets))
            for j, t in enumerate(self.targets):
                v = np.log(self.Pv[j])
                for w, a in enumerate(sample):
                    if self.is_continuous[w]:
                        mu = self.Pav[t][w][0]
                        sigma = self.Pav[t][w][1]
                        if sigma != 0:
                            v += log_gaussian(float(a), mu, sigma)
                    else:
                        if a == -1:
                            a_key = 0  
                        else:
                            a_key = a
                        v += np.log(self.Pav[t][w].get(a_key, 1e-10))
                p[j] = v
            Z.append(self.targets[np.argmax(p)])
        return Z

if __name__ == "__main__":
    data = {
        "outlook": ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", 
                    "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", 
                    "overcast", "rainy"],
        "temp": ["hot", "hot", "hot", "mild", "cool", "cool", "cool", 
                "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
        "humidity": ["high", "high", "high", "high", "normal", "normal", "normal",
                    "high", "normal", "normal", "normal", "high", "normal", "high"],
        "windy": [False, True, False, False, False, True, True,
                False, False, False, True, True, False, True],
        "play": ["no", "no", "yes", "yes", "yes", "no", "yes",
                "no", "yes", "yes", "yes", "yes", "yes", "no"]
    }

    X = np.array([data["outlook"], data["temp"], data["humidity"], data["windy"]]).T
    y = np.array(data["play"])
    is_continuous = [False, False, False, True] 

    model = NaiveBayes()
    model(X, y)
    model.train()

    test_data = {
        "outlook": ["sunny", "rainy", "overcast"],
        "temp": ["mild", "cool", "hot"],
        "humidity": ["high", "normal", "normal"],
        "windy": [True, False, True]
    }
    X_test = np.array([test_data["outlook"], test_data["temp"], 
                    test_data["humidity"], test_data["windy"]]).T

    pred = model.test(X_test)
    print(f"Predictions for play tennis: {pred}")