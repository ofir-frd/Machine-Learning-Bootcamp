import numpy as np
import os
from imageio import imread
from imageio import imwrite
from sklearn.decomposition import PCA


# import pandas as pd


# reads the first image from each folder and create a 2D array with each column per image
# return: 2D arrays with images data
def read_images():
    
    face = np.ones((108000, 1), dtype=int)
    for root, dirs, files in os.walk('faces94\\', topdown=True):
        for name in files:
            full_path = os.path.join(root, name)
            # print(full_path)
            face = np.column_stack((face, (np.reshape(imread(full_path), (-1, 1)))))

            break

    face = np.delete(face, 0, 1)

    return face


# reshape and write edited images
def write_images(faces: np.ndarray):

    output_folder_base = 'results\\'

    i = 0

    for face in faces.T:
        output_folder = output_folder_base + str(i) + '.jpg'
        imwrite(output_folder, np.reshape(face, (200, 180, 3)))
        i += 1


# Principal component analysis
def my_pca(faces: np.ndarray):

    reduced_faces = np.ones((108000, 1), dtype=int)

    # calculate and subtract mean per image
    mean_matrix = [np.mean(face) for face in faces.T]  # subtract mean per image
    x = faces - mean_matrix

    # covariance
    s = np.cov(x.T)

    # calculate eigenvalue & eigenvector
    eigen_values, eigen_vectors = np.linalg.eig(s)

    # todo: calculate total energy and remove eigen values

    return reduced_faces * eigen_values


def main():
    # Create dataset
    faces = read_images()

    #%% PCA with sklearn library
    pca = PCA()
    pca.fit(faces)
    reduced_faces = pca.transform(faces)
    write_images(reduced_faces)

    #%% PCA with numpy
    reduced_faces = my_pca(faces)
    write_images(reduced_faces)


if __name__ == '__main__':
    main()
