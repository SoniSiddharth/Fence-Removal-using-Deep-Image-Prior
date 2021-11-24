import re
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import DBSCAN

def unit_vector(vector):
    """ normalizes a given vector  """
    return vector / np.linalg.norm(np.array(vector))

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def yes_or_no(question):
    """ Poses a question to the user and waits for either a y/n """
    while "the answer is invalid":
        reply = str(input(question+' (y/n) ')).lower().strip()
        if reply[0] == "y":
            return True
        if reply[0] == "n":
            return False


def get_filename_parts(inputfile):
    """cut the fname string into 3 parts: path+name+extension """
    extension_start = [m.start() for m in re.finditer("\.",inputfile)][-1]
    try:
        fname_start = [m.start() for m in re.finditer("/",inputfile)][-1]
        path = inputfile[:fname_start+1]
    except IndexError:
        fname_start=-1
        path="/"
    fname = inputfile[fname_start+1:extension_start]
    extension = inputfile[extension_start:]
    return path,fname, extension


def getXYSfromKPS(kps):
    """get the attributes corresponding to x/y-coordinates and sizes of the keypoints kps """
    n_kp = np.size(kps)
    x = np.zeros(n_kp)
    y = np.zeros(n_kp)
    sizes = np.zeros(n_kp)
    for i in range(n_kp):
        x[i],y[i] = kps[i].pt
        sizes[i] = kps[i].size
    return np.array(x),np.array(y),np.array(sizes)


def plot_clusters(img,kps,labels,colormap='jet'):
    """plot all keypoints with color according to their label on top of the image """
    #get coordinates and sizes of all keypoints
    x,y,sizes = getXYSfromKPS(kps)

    #plot background image
    

    cmap = plt.get_cmap(colormap,len(set(labels)))
    plot_img = img
    for i in range(len(set(labels))):
        c = np.array(cmap(i))*255
        for k in range(np.size(kps)):
            if labels[k]==i:
                plot_img = cv.circle(plot_img,(int(x[k]),int(y[k])),int(sizes[k]),c)
    return plot_img

def plot_one_cluster(img,kps,labels,cluster_label,colormap="jet"):
    #get coordinates and sizes of all keypoints
    x,y,sizes = getXYSfromKPS(kps)

    #plot background image
    cmap = plt.get_cmap(colormap,len(set(labels)))
    plot_img = img
    c = np.array(cmap(cluster_label))*255
    for k in range(np.size(kps)):
        if labels[k]==cluster_label:
            plot_img = cv.circle(plot_img,(int(x[k]),int(y[k])),int(sizes[k]),c)
    return plot_img

def kNearestNeighbours(x,k,eps = 0):
    '''
    function to find and store the difference vectors to the 
    k nearest neighbors for all x's in a list
    Parameters:
    -----------
    inputs:
    x   - an array with shape (N,m) where N is number of points and m is dimension
    k   - number of nearest neighbors to calculate difference vectors to
    outputs:
    KNN - return array (dim(x)*k,2) with distances to the k nearest neighbours from any x
    '''
    N,m = np.shape(x)
    if k >= N:
        #check if k is bigger or equal to the number of points
        print("k too big for array of size {}".format(N))
        print("setting k to {}-1".format(N))
        k = N-1
    kNN = np.array([])
    for i in range(N):
        #for every x check 6 NN that are nonzero    
        dist = np.linalg.norm(-x+x[i,:],axis=1)
        sort_i = np.argsort(dist)
        dist = np.take_along_axis(dist,sort_i,axis=0)
        Nzeros = np.size(dist[dist<=eps])
        for j in range(k):
            kNN = np.append(kNN,x[sort_i[Nzeros+j],:]-x[i,:])
    kNN = kNN.reshape((N*k,m))
    return kNN

def plot_clustered_NNVs(kNN_right,kNN_labels,kNN_red,a,b):
    plt.axis("equal")
    for i in set(kNN_labels):
        plt.scatter(kNN_right[kNN_labels == i,0],-kNN_right[kNN_labels == i,1],s = 1)
    plt.scatter(kNN_red[:,0],-kNN_red[:,1],label = "reduced",s = 2)
    plt.arrow(0,0,a[0],-a[1])
    plt.arrow(0,0,b[0],-b[1])

def DensityClustering(x,rtol,mins=3):
    '''
    Use DBSCAN algorithm to cluster a (potentially noisy) list of vectors
    Parameters:
    ----------
    Input
    x - list of vectors
    rtol    - distance measure used for the DBSCAN clustering
    mins - minimum amount of samples to form cluster in DBSCAN
    Output:
    unique - average vectors of each cluster
    labels - labels for each vector
    '''
    N,m = np.shape(x)
    clustering = DBSCAN(eps = rtol,min_samples = mins).fit(x)
    labels = clustering.labels_

    unique = np.array([])
    for i in set(labels):
        if i != -1:
            unique = np.append(unique,np.average(x[labels==i,:],axis=0))
    unique = unique.reshape((np.size(unique)//m,m))
    
    return unique,labels