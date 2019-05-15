# MNIST in TensorFlow

This directory builds a convolutional neural net to classify the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/) using the
[tf.data](https://www.tensorflow.org/api_docs/python/tf/data),
[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator),
and
[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers)
APIs.

# Gradient Setup

## Single Node Training on Gradient

### Install Gradient CLI

```
pip install paperspace
```

[Please check our documentation on how to install Gradient CLI and obtain a Token](https://app.gitbook.com/@paperspace/s/gradient/cli/install-the-cli)

### Create project and obtain its handle

[Please check our documentation on how to create a project](https://app.gitbook.com/@paperspace/s/gradient/projects/create-a-project)

### Create and start single node experiment

```
paperspace-python experiments createAndStart singlenode --name mnist --projectHandle <your project handle> --experimentEnv "{\"EPOCHS_EVAL\":5,\"TRAIN_EPOCHS\":10,\"MAX_STEPS\":1000,\"EVAL_SECS\":10}" --container tensorflow/tensorflow:1.13.1-gpu-py3 --machineType K80 --command "python mnist.py" --workspaceUrl https://github.com/Paperspace/mnist-sample.git
```

That's it!

## Multinode Training on Gradient

### Create and start distributed multinode experiment

```
paperspace-python experiments createAndStart multinode --name mnist-multinode --projectHandle <your project handle> --experimentEnv "{\"EPOCHS_EVAL\":5,\"TRAIN_EPOCHS\":10,\"MAX_STEPS\":1000,\"EVAL_SECS\":10}" --experimentTypeId GRPC --workerContainer tensorflow/tensorflow:1.13.1-gpu-py3 --workerMachineType K80 --workerCommand 'pip install -r requirements.txt && python mnist.py' --workerCount 2 --parameterServerContainer tensorflow/tensorflow:1.13.1-py3 --parameterServerMachineType K80 --parameterServerCommand 'pip install -r requirements.txt && python mnist.py' --parameterServerCount 1 --workspaceUrl https://github.com/Paperspace/mnist-sample.git
```

### Modify your code to run distributed on Gradient

#### Set `TF_CONFIG` environment variable

First import from gradient-sdk:

```
from gradient_sdk import get_tf_config
```

then in your main():

```
if __name__ == '__main__':
    get_tf_config()
```

This function will set `TF_CONFIG`, `INDEX` and `TYPE` for each node.

For multi-worker training, as mentioned before, you need to set the `TF_CONFIG` environment variable for each binary running in your cluster. The `TF_CONFIG` environment variable is a JSON string that specifies the tasks that constitute a cluster, each task's address, and each task's role in the cluster.

### Exporting a Model for deployments

#### Export your Tensorflow model

In order to serve a Tensorflow model, simply export a SavedModel from your Tensorflow program. [SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md) is a language-neutral, recoverable, hermetic serialization format that enables higher-level systems and tools to produce, consume, and transform TensorFlow models.

Please refer to [Tensorflow documentation](https://www.tensorflow.org/guide/saved_model#save_and_restore_models) for detailed instructions on how to export SavedModels.

#### Example code showing how to export your model:

```
tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

#Starting to Export model
image = tf.placeholder(tf.float32, [None, 28, 28])
input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
mnist_classifier.export_savedmodel(<export directory>,
                                    input_fn,
                                    strip_default_attrs=True)
#Model Exported
```

We use TensorFlow's [SavedModelBuilder module](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py) to export the model. SavedModelBuilder saves a "snapshot" of the trained model to reliable storage so that it can be loaded later for inference.

For details on the SavedModel format, please see the documentation at [SavedModel README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).

For export directory, be sure to set it to `PS_MODEL_PATH` when running a model deployment on Gradient:

```
export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH'))
```

You can also use Gradient SDK to ensure you have the correct path:

```
from gradient_sdk.utils import data_dir, model_dir, export_dir
```

# Local Setup

To begin, you'll simply need the latest version of TensorFlow installed.

First make sure you've [added the models folder to your Python path](/official/#running-the-models); otherwise you may encounter an error like `ImportError: No module named mnist`.

Then, to train the model, simply run:

```
python mnist.py
```

###(Optional)Running localy from your virtualenv
You can download mnist sample on your local machine and run it on your computer.

- download code from github:
```
git clone git@github.com:Paperspace/mnist-sample.git
```
- create virtual environment (we recommend using python3) and activate it
```
cd mnist-sample
```
```
python3 -m venv venv
```
```
source venv/bin/activate
```

- install local requirements

```
pip install -r requirements-local.txt
```

- run model

Before you run it you should know that it will be running for a long time.
Command to run model 
```
python mnist.py
```
If you want to shorten model training time you can change max steps parameter.
```
python mnist.py --max_steps=1500
```

Mnist data are downloaded to '/data' directory.

Model results are stored to '/models' directory.

Both directories can be safely deleted after interpreting model results.

## Exporting the model

You can export the model into Tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model) format by using the argument `--export_dir`:

```
python mnist.py --export_dir /tmp/mnist_saved_model
```

## Training the model for use with Tensorflow Serving on a CPU

If you are training on Tensorflow using a GPU but would like to export the model for use in Tensorflow Serving on a CPU-only server, you can train and/or export the model using `--data_format=channels_last`:

```
python mnist.py --data_format=channels_last
```

The SavedModel will be saved in a timestamped directory under `/tmp/mnist_saved_model/` (e.g. `/tmp/mnist_saved_model/1513630966/`).

## Getting predictions with SavedModel

Use [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel) to inspect and execute the SavedModel.
