# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, _ = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
            paths = paths[0:10]

            # Load the model
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]
        
            emb_array = np.zeros((2,embedding_size))
            # Run forward pass to calculate embeddings
            print('First batch size')
            paths_batch = paths[0:1]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            emb_array[0,:] = emb[0,:]
            #print(emb_array[0,:])
        
            print('Second batch size')
            paths_batch = paths[0:200]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            emb_array[1,:] = emb[0,:]
            #print(emb_array[1,:])
            
            print(emb_array[1,:]-emb_array[0,:])
            
        
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
