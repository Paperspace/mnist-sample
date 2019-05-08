# MNIST in TensorFlow

This directory builds a convolutional neural net to classify the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/) using the
[tf.data](https://www.tensorflow.org/api_docs/python/tf/data),
[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator),
and
[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers)
APIs.

# Gradient Setup
## Single Training on Gradient

### Install Gradient CLI

```
pip install paperspace
```
[Please check our documentation on how to install CLI and obtain Token](https://app.gitbook.com/@paperspace/s/gradient/cli/install-the-cli)

### Create project and obtain its handle

[Please check our documentation on how to create a project]
(https://app.gitbook.com/@paperspace/s/gradient/projects/create-a-project)

### Create and start single node experiment

```
paperspace-python experiments createAndStart singlenode --name mnist --projectHandle <your project handle> --experimentEnv "{\"EPOCHS_EVAL\":5,\"TRAIN_EPOCHS\":10,\"MAX_STEPS\":1000,\"EVAL_SECS\":10}" --container tensorflow/tensorflow:1.13.1-gpu-py3 --machineType K80 --command "python mnist.py" --workspaceUrl https://github.com/Paperspace/mnist-sample.git
```

Thats it!

### Create and start distributed multinode experiment

```
paperspace-python experiments createAndStart multinode --name mnist-multinode --projectHandle <your project handle> --experimentEnv "{\"EPOCHS_EVAL\":5,\"TRAIN_EPOCHS\":10,\"MAX_STEPS\":1000,\"EVAL_SECS\":10}" --experimentTypeId GRPC --workerContainer tensorflow/tensorflow:1.13.1-gpu-py3 --workerMachineType K80 --workerCommand "python mnist.py" --workerCount 2 --parameterServerContainer tensorflow/tensorflow:1.13.1-py3 --parameterServerMachineType K80 --parameterServerCommand "python mnist.py" --parameterServerCount 1 --workspaceUrl https://github.com/Paperspace/mnist-sample.git
```

# Local Setup

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
