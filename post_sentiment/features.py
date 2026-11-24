from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True)),
        ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=18000, sublinear_tf=True)),
    ])
