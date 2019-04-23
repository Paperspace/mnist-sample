# MNIST in TensorFlow

This is random, minor change to try triggering an experiment.
This directory builds a convolutional neural net to classify the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/) using the
[tf.data](https://www.tensorflow.org/api_docs/python/tf/data),
[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator),
and
[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers)
APIs.


## Setup

To begin, you'll simply need the latest version of TensorFlow installed.
First make sure you've [added the models folder to your Python path](/official/#running-the-models); otherwise you may encounter an error like `ImportError: No module named mnist`.

Then to train the model, run the following:

```
python mnist.py
```

Distributed Training on Gradient

Just run the example with following parameters:
```
  "name": "Mnist Sample",
  "projectHandle": "<your project handle",
  "parameterServerContainer": "tensorflow/tensorflow:1.13.1-gpu-py3",
  "parameterServerMachineType": "K80",
  "parameterServerCount": 1,
  "workerCommand": "python mnist.py",
  "workerContainer": "tensorflow/tensorflow:1.13.1-gpu-py3",
  "workspaceUrl": "git+https://github.com/paperspace/mnist-sample.git",
  "workerMachineType": "K80",
  "workerCount": 2,
  "parameterServerCommand": "python mnist.py"
```
Gradient will generate TF_CONFIG in base64 format for each node so all you need to do in your other projects:
```
paperspace_tf_config = json.loads(base64.urlsafe_b64decode(os.environ.get('TF_CONFIG')).decode('utf-8'))
```

## Exporting the model

You can export the model into Tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model) format by using the argument `--export_dir`:

```
python mnist.py --export_dir /tmp/mnist_saved_model
```

## Training the model for use with Tensorflow Serving on a CPU

If you are training on Tensorflow using a GPU but would like to export the model for use in Tensorflow Serving on a CPU-only server you can train and/or export the model using ` --data_format=channels_last`:
```
python mnist.py --data_format=channels_last
```

The SavedModel will be saved in a timestamped directory under `/tmp/mnist_saved_model/` (e.g. `/tmp/mnist_saved_model/1513630966/`).

**Getting predictions with SavedModel**
Use [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel) to inspect and execute the SavedModel.
