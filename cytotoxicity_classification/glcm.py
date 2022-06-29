from skimage.feature import graycomatrix, graycoprops
import numpy as np


def glcm_features(image, features=["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]):
    m_glcm = graycomatrix(image, distances=[2, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          symmetric=True, normed=True)
    feature_dict = {}
    for feature in features:
        feature_dict[feature] = graycoprops(m_glcm, feature).tolist()
    return feature_dict
