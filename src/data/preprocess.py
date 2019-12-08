"""
    Preprocess data ready for feeding to the model.
"""
from sklearn.decomposition import PCA
from typing import Dict, Tuple

def run_pca(dataset: Dict) -> Tuple[Dict, PCA]:
    """Apply PCA to a dataset"""

    pca = PCA(n_components='mle')

    transformed_data = pca.fit_transform(dataset['data'])

    result = {
        'data': transformed_data,
        'target': dataset['target']
    }

    return result, pca



