# Simple TFServing example; Based on 
# https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py
# Added simpler mnist loading parts and removed some complexity


"""A client that talks to tensorflow_model_server loaded with mnist model.
The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.
Typical usage example:
    mnist_client.py --num_tests=100 --server=localhost:8500
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

import grpc
import numpy
import tensorflow as tf
import logging
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras.preprocessing import image
import numpy as np
from keras.datasets import mnist
import time
from random import randint

logging.disable(logging.WARNING)
tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 1, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

_counter = 0
_start = 0
def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    global _counter 
    global _start
    exception = result_future.exception()
    if exception:
      print(exception)
    else:
      #print("From Callback",result_future.result().outputs['probabilities'])
      if(_start == 0):
          _start = time.time()
      response = numpy.array(
          result_future.result().outputs['probabilities'].float_val)
      if response.size:
          prediction = numpy.argmax(response)
          if( (_counter % 10) ==0):
              print("[", _counter,"] From Callback Predicted Result is ", prediction,"confidence= ",response[prediction])
      else:
          print("empty response")
      _counter += 1
      if (_counter == FLAGS.num_tests):
          end = time.time()
          print("Time for ",FLAGS.num_tests," is ",end -_start)


def do_inference(hostport, work_dir, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.
    Args:
        hostport: Host:port address of the PredictionService.
        work_dir: The full path of working directory for test data set.
        concurrency: Maximum number of concurrent requests.
        num_tests: Number of test images to use.
    Returns:
        The classification error rate.
    Raises:
        IOError: An error occurred processing test data set.
    """
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'serving_default'
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    # For loading images
    # img = image.load_img('./data/mnist_png/testing/0/10.png', target_size=(28,28))
    #x = image.img_to_array(img)
    #request.inputs['images'].CopyFrom(
    #tf.contrib.util.make_tensor_proto(image2, shape=[1,1,image2.size]))
    image_index=randint(0, 9999)
    x= X_train[image_index][0]
    print("Shape is ",x.shape," Label is ", y_train[image_index])
    start = time.time()
    for _ in range(num_tests):
        xt= x.astype(np.float32)
        request.inputs['image'].CopyFrom(tf.contrib.util.make_tensor_proto(xt, shape=[1,1,28, 28]))
        #result_counter.throttle()
        result_future = stub.Predict.future(request, 10.25)  # 5 seconds  
        result_future.add_done_callback(_callback)
        end = time.time()
        image_index=randint(0, 9999)
        x= X_train[image_index][0]
    print("Time to Send ", num_tests ," is ",end - start)
        
    time.sleep(6)
    # if things are wrong the callback will not come, so uncomment below to force the result
    # or get to see what is the bug
    #res= result_future.result()
    #response = numpy.array(res.outputs['probabilities'].float_val)
    #prediction = numpy.argmax(response)
    #print("Predicted Result is ", prediction,"Detection Probability= ",response[prediction])
  

def main(_):
  if FLAGS.num_tests > 20000:
    print('num_tests should not be greater than 20k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
                            FLAGS.concurrency, FLAGS.num_tests)  

if __name__ == '__main__':
    print ("hello from TFServing client slim")
    tf.compat.v1.app.run()

