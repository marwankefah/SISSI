import numpy as np
from tqdm import tqdm


class Classifier:
    def __init__(self):
        # Determines if sample shall be classified as -1 or 1 given threshold
        self.polarity = 1
        # The index of the feature used to make classification
        self.feature_index = None
        # The threshold value that the feature should be measured against
        self.threshold = None
        # Value indicative of the classifier's accuracy
        self.alpha = None

    def predict(self, X):
        if self.polarity == 1:
            return X <= self.threshold
        else:
            return X >= self.threshold


class AdaBoostSelection:
    def __init__(self, T):
        """
        Called when initializing the classifier
        """
        self.T = T
        self.ChosenFeatures = None
        self.Classifiers = []

    def fit(self, X, Y):
        w = np.ones((X.shape[0], 1))
        w[Y == 1] = 1 / (2 * np.sum(Y == 1))
        w[Y == 0] = 1 / (2 * np.sum(Y == 0))
        m = np.sum(Y == 1)
        l = np.sum(Y == 0)
        N = X.shape[1]
        self.ChosenFeatures = np.zeros(N)
        # TODO check if threshold depends on weight

        Y = Y.reshape(Y.shape[0])
        for i in tqdm(range(self.T)):
            # TODO change thresholding in the future to parity !!!
            classifier = Classifier()
            threshpos = np.sum(w[Y == 1] * X[Y == 1],
                               axis=0) / np.sum(w[Y == 1])
            threshneg = np.sum(w[Y == 0] * X[Y == 0],
                               axis=0) / np.sum(w[Y == 0])
            threshold = 0.5 * (threshpos + threshneg)
            p = np.ones((X.shape[1]))
            p[threshneg < threshpos] = -1
            # Parity for comparison
            h = p * X <= p * threshold
            error = np.sum(
                w * np.abs(h - Y.reshape(Y.shape[0], 1)), axis=0) / np.sum(w)
            error[self.ChosenFeatures == 1] = 1

            # sorting errors in ascending order and getting the indices sorted
            chosen = np.argsort(error)
            self.ChosenFeatures[chosen[0]] = 1
            chosenerror = error[chosen[0]]
            alpha = np.log((1 - chosenerror) / chosenerror)
            w = w * np.exp(alpha * h[:, chosen[0]] != Y).reshape(w.shape)
            classifier.feature_index = chosen[0]
            classifier.polarity = p[chosen[0]]
            classifier.alpha = alpha
            classifier.threshold = threshold[chosen[0]]
            self.Classifiers.append(classifier)
            w /= np.sum(w)

    def predict(self, X):
        predictionSum = np.zeros((X.shape[0], 1))
        sum = 0
        for classifier in self.Classifiers:
            predictionSum += (classifier.alpha * classifier.predict(X[:, classifier.feature_index])).reshape(
                predictionSum.shape)
            sum += classifier.alpha
        sum = sum / 2
        predictionSum[predictionSum < sum] = 0
        predictionSum[predictionSum > sum] = 1
        return predictionSum

    def get_params(self):
        return {"Features": self.ChosenFeatures, "Classifiers": self.Classifiers}

    def validate(self, X, Y):
        h = self.predict(X)
        return np.sum(h == Y)/X.shape[0]
