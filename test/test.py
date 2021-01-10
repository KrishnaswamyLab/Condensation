import numpy as np
import diffusion_condensation as dc
import warnings
import parameterized

warnings.simplefilter("error")


@parameterized.parameterized([(None,), (100,)])
def test(partitions, landmarks):
    X = np.random.normal(0, 1, (200, 200))
    dc_op = dc.Diffusion_Condensation(
        landmarks=landmarks,  # n_pca=20
    )
    NxTs = dc_op.fit_transform(X)
    return
    assert len(NxTs[0]) == X.shape[0], (X.shape, len(NxTs))

    Y = np.random.normal(0.5, 1, (200, 200))
    dc_op = dc.Diffusion_Condensation(
        landmarks=landmarks,  # n_pca=20
    )
    NxTs = dc_op.fit_transform(Y)
    return
    assert len(NxTs[0]) == Y.shape[0], (Y.shape, len(NxTs))

    tree = mp_op.visualize_homology()
    tree_clusters = mp_op.get_homology_clusters(NxTs[-10])

    assert tree.shape[0] == len(tree_clusters)
