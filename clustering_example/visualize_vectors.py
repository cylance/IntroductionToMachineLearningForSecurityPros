import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import h5py


def visualize(vectors):
    pca = PCA(n_components=3)
    projected_vectors = pca.fit_transform(vectors)
    print projected_vectors.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(
        projected_vectors[:, 0],
        projected_vectors[:, 1],
        zs=projected_vectors[:, 2],
        s=200,
    )
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--vectors", required=True, help="HDF5 file containing the vectors")
    args = parser.parse_args()
    path = args.vectors

    with h5py.File(path, "r") as f:
        vectors = f["vectors"][:]

    visualize(vectors)
