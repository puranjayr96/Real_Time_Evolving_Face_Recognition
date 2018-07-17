from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import detect_face
import os
import sys
import math
import pickle
from sklearn.svm import SVC


def test(emb,labels,class_names):
    with open('def_embeddings.pkl', 'rb') as fi:
        (def_emb_arrays, def_labels) = pickle.load(fi)


    udir = os.path.expanduser('cap_embeddings.pkl')
    if os.path.exists(udir):
        with open('cap_embeddings.pkl', 'rb') as cein:
            cap_embeddings,cap_labels = pickle.load(cein)
            cap_embeddings = np.concatenate((cap_embeddings,emb),axis = 0)
            cap_labels.extend(labels)
            print('Cap Accessed')

    else:
        cap_labels = labels
        cap_embeddings = emb
        print('Cap not accessed')

    with open('cap_embeddings.pkl','wb') as ceout:
        pickle.dump((cap_embeddings,cap_labels),ceout)
    def_emb_arrays = np.concatenate((def_emb_arrays, cap_embeddings), axis=0)
    def_labels.extend(cap_labels)


    with tf.Graph().as_default():

        with tf.Session() as sess:
            classifier_filename = './my_class/my_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)


            # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(def_emb_arrays, def_labels)

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            print('Goodluck')