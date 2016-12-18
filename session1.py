# Import libraries
import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA
import random 


# Function: train_features()
# Description:
# Input: train_images_filenames, train_labels, extractor, n_images, isPCA=True
# Output: D, L
def train_features(images_filenames, labels, textractor, n_images, is_pca):

    # SIFTdetector = cv2.SIFT(nfeatures=n_Sift_Features)
    # read the just 30 train images per class
    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays

    train_descriptors = []
    train_label_per_descriptor = []

    for i in range(len(images_filenames)):
        filename = images_filenames[i]
        if train_label_per_descriptor.count(labels[i]) < n_images:
            print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = textractor.detectAndCompute(gray, None)
            train_descriptors.append(des)
            train_label_per_descriptor.append(labels[i])
            print str(len(kpt)) + ' extracted keypoints and descriptors'

    d = train_descriptors[0]
    l = np.array([train_label_per_descriptor[0]] * train_descriptors[0].shape[0])

    for i in range(1, len(train_descriptors)):
        d = np.vstack((d, train_descriptors[i]))
        l = np.hstack((l, np.array([train_label_per_descriptor[i]] * train_descriptors[i].shape[0])))

    if is_pca:
        print "Apply PCA algorithm to reduce dimensionality"
        pca.fit(d)
        pca.transform(d)
    return d, l


# Function: train_SVM()
# Description:
# Input: d_scaled, l, kernel_type
# Output: clf
def train_svm(d_scaled, l, kernel_type):
    print 'Training the SVM classifier...'
    clf_train = svm.SVC(kernel=kernel_type, C=100).fit(d_scaled, l)
    print 'Done!'
    return clf_train


# Function: test_classifier()
# Description:
# Input: images_filenames, labels, extractor, clf, stdSlr, pca, isPCA=True
# Output: numcorrect
def classifier(images_filenames, labels, cextractor, cclf, cstdslr, is_pca, cpca,):
    numtestimages = 0
    cnumcorrect = 0
    for i in range(len(images_filenames)):
        filename = images_filenames[i]
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)

        kpt, des = cextractor.detectAndCompute(gray, None)
        if is_pca:
            cpca.transform(des)

        predictions = cclf.predict(cstdslr.transform(des))
        values, counts = np.unique(predictions, return_counts=True)
        predictedclass = values[np.argmax(counts)]
        print 'image ' + filename + ' was from class ' + labels[i] + ' and was predicted ' + predictedclass
        numtestimages += 1
        if predictedclass == labels[i]:
            cnumcorrect += 1
    return cnumcorrect

# Function: test_classifier_pseudorandom()
# Description: Gives another criterion for assigning the label to the test sample
# Input: images_filenames, labels, extractor, clf, stdSlr, pca, isPCA=True
# Output: numcorrect
def test_classifier_pseudorandom(test_images_filenames, test_labels, extractor, clf, stdSlr, pca, isPCA = True):

    numtestimages=0
    numcorrect=0
    for i in range(len(test_images_filenames)):
        filename=test_images_filenames[i]
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        
        kpt,des=extractor.detectAndCompute(gray,None)
        if isPCA:
            pca.transform(des)

        predictions = clf.predict(stdSlr.transform(des))
        values, counts = np.unique(predictions, return_counts=True)
        valids = []
        for x in range(len(values)):
            if abs(counts[x] - max(counts)) <= float(max(counts)) / counts.mean():
                valids.append(values[x])
                for j in range(counts[x]/(abs(counts[x] - max(counts))+1)):
                    valids.append(values[x])

        random.shuffle(valids)
        predictedclass = valids[random.randint(0, len(valids) - 1)]
        print 'image '+ filename +' was from class '+test_labels[i] + ' and was predicted '+predictedclass
        numtestimages+=1
        if predictedclass==test_labels[i]:
            numcorrect+=1
    return numcorrect

# ------------------ Main function ------------------ #
if __name__ == "__main__":

    # Start timer to analyse performance
    start = time.time()

    # Read the train and test files
    train_images_filenames = cPickle.load(open('dat_files/train_images_filenames.dat', 'r'))
    test_images_filenames = cPickle.load(open('dat_files/test_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('dat_files/train_labels.dat', 'r'))
    test_labels = cPickle.load(open('dat_files/test_labels.dat', 'r'))

    print 'Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels)
    print 'Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels)

    # PCA components
    pca = PCA(n_components=20)

    # Create the SIFT detector object
    extractor = cv2.SIFT(nfeatures=100)

    # Create the SURF detector object
    # extractor = cv2.SURF(nfeaures = 100)

    num_images = 15
    apply_pca = True
    D, L = train_features(train_images_filenames, train_labels, extractor, num_images, apply_pca)

    # Train a linear SVM classifier
    stdSlr = StandardScaler().fit(D)
    D_scaled = stdSlr.transform(D)

    kernel = 'linear'
    clf = train_svm(D_scaled, L, kernel)

    print 'Classifier trained'

    # Get all the test data and predict their labels
    numcorrect = classifier(test_images_filenames, test_labels, extractor, clf, stdSlr, True, pca)

    print 'Final accuracy: ' + str(numcorrect * 100.0 / len(test_images_filenames))

    # End timer to print time spent
    end = time.time()
    print 'Done in ' + str(end - start) + ' secs.'
