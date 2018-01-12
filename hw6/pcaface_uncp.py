from skimage import io, transform
# import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import sys

IMG_PATH = sys.argv[1]
IMG_NAME = sys.argv[2]
LENGTH = 600
SHAPE_OF_IMAGE = (LENGTH, LENGTH, 3)
NUM_EFACE = 4


def load_pkl(path):
    file = open(path, 'rb')
    obj = pkl.load(file)
    file.close()
    return obj

def save_pkl(path, obj):
    file = open(path, 'wb')
    pkl.dump(obj, file)
    file.close()

def loadDataset():
    face_ls = []
    imgname_ls = os.listdir(path=IMG_PATH)
    imgname_ls.sort() 
    for imgname in imgname_ls:
        img = io.imread(os.path.join(IMG_PATH, imgname))
        face_ls.append(img)

    return face_ls

def preprocess(listOfFaces):
    flatten_ls = []
    for face in listOfFaces:
        compressed = face
        # compressed = transform.resize(face, SHAPE_OF_IMAGE)
        # print(compressed.shape)
        flatten_ls.append(compressed.flatten())
        # print('-------------------------prep-----------------------------')
        # print(compressed.flatten())
        # print('-------------------------prep-----------------------------')
        # flatten_ls.append(face.flatten())
        # print((face.flatten()).shape)
    X_mean = np.mean(np.array(flatten_ls))
    return (np.array(flatten_ls)), X_mean # one row per face

def SVD(X, X_mean, UFilename, sFilename, VFilename):
    X = X.T
    # print(r_mean)
    U, s, V = np.linalg.svd(X-X_mean, full_matrices=False)
    # save_pkl(UFilename, U)
    # save_pkl(sFilename, s)
    # save_pkl(VFilename, V)
    return U, s, V

def genEigenface(U, numOfEigenfaces=4):
    # print(rU.shape)
    ef = np.zeros(shape=(numOfEigenfaces, U.shape[0]))
    for i in range(numOfEigenfaces):
        ef[i] = (U.T)[i]
    return ef # one row per eigenface


# def drawEigenface(eigenface):
#     fig = plt.figure()
#     print(eigenface.shape)
#     for i in range(eigenface.shape[0]):
#         tmp = eigenface[i]
#         tmp -= np.min(tmp)
#         tmp /= np.max(tmp)
#         tmp = (tmp*255).astype(np.uint8)
#         tmp = tmp.reshape(SHAPE_OF_IMAGE)
#         ax = fig.add_subplot(2,2,i+1)
#         ax.imshow(tmp)
#         plt.xticks(np.array([]))
#         plt.yticks(np.array([]))
#         plt.tight_layout()
    
#     fig.savefig('./tmp0112/eigenfaces'+str(eigenface.shape[0])+'.png')



def reconstruct(imagename, eigenface, X_mean):
    face = io.imread(os.path.join(IMG_PATH, imagename))
    # face = transform.resize(face, SHAPE_OF_IMAGE)
    face = face.flatten()
    # print(face)
    # print(X_mean)
    # print(face-X_mean)
    reconst_face = np.zeros(eigenface[0].shape) 
    tmp = []      
    for j in range(eigenface.shape[0]):
        y = face - X_mean
        weight = 0
        weight = np.dot(y, eigenface[j])
        reconst_face += weight*eigenface[j]
        # print(eigenface[j])
        tmp.append(weight)
        # print(weight)
    reconst_face += X_mean
    # print('-------------------------weight-----------------------------')
    # print(tmp)
    # print('-------------------------weight-----------------------------')
    tmp = reconst_face
    tmp -= np.min(tmp)
    tmp /= np.max(tmp)
    tmp = (tmp*255).astype(np.uint8)
    tmp = tmp.reshape(SHAPE_OF_IMAGE)
    io.imsave('./reconstruction.jpg',tmp)
    return 0



def main():
    face_ls = loadDataset()
    face, X_mean = preprocess(face_ls)
    # print('X_mean', X_mean)

    U, s, V = SVD(face, X_mean, './svd_pkl/UU.pkl', './svd_pkl/ss.pkl', './svd_pkl/VV.pkl')
    # U = load_pkl('./svd_pkl/100U.pkl')
    eface = genEigenface(U, NUM_EFACE)
    # drawEigenface(eface)    
    reconstruct(IMG_NAME, eface, X_mean)
    
    return 0


if __name__ == '__main__':
    main()