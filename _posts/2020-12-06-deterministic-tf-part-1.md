---
title: "Deterministic Tensorflow Part 1: Model Training"
date: 2020-12-06 11:00:00 +1000
image: /assets/img/posts/pre-emptible-venn.jpg
categories: [Tensorflow, Data]
tags: [pipeline, performance, parallel, deterministic, pre-emptible]     # TAG names should always be lowercase
---

Reproducibility is critical to any scientific endeavour, and machine learning is no exception. Releasing code that generates results from papers is an important step in addressing this, but difficulties arise in random aspect of neural network training including data shuffling, augmentation and network initialization, making exact replication of results difficult. Two common approaches for handling these difficulties are:

1. repeating experiments multiple times and reporting statistics; and
2. managing the random state.

This post looks at the second point, particularly as it applies to training [tensorflow](https://tensorflow.org)'s [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)s. We'll be focusing on two properties of our programs:

- _Determinism_: our programs should produce exactly the same outputs for the same inputs.
- _Pre-emptibility_: our programs should be able to be interrupted and restarted without affecting the results.

Note that just because our programs are deterministic doesn't mean there aren't sources of pseudo-randomness - just that those sources need to be configurable such that they can perform every time. This is commonly done by setting a program's random seed, but as we'll see that's not necessarily the end of the story - particularly if we want our programs to be pre-emptible. For more information about setting random seeds and random number generation in tensorflow check out tensorflow's [random numbers guide](https://www.tensorflow.org/guide/random_numbers).

This is part 1 of a 2-part series looking at deterministic, pre-emptible tensorflow. [Part 2](../deterministic-tf-part-2) takes a deep dive into data augmentation.

## Modules, Checkpoints and BackupAndRestore

Before we get into the specifics of training deterministic pre-emptible models, it's important that we understand the mechanism by which we'll be saving and restoring our training state. We'll be using 2 key classes provided in tensorflow:

- [tf.Module](https://www.tensorflow.org/api_docs/python/tf/Module): base class for objects that track dependencies, where dependencies are defined as savable objects assigned as attributes. Most public `tf.keras` classes including `Model` and `Layer` subclass this.
- [tf.train.Checkpoint](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint): for saving and restoring `Module`s, including any tracked dependencies.

The following example shows simple usage.

```python
import tensorflow as tf


class Foo(tf.Module):
    def __init__(self):
        self.bar = tf.Variable(0, dtype=tf.int64)


foo = Foo()
foo.bar.assign(2)
chkpt = tf.train.Checkpoint(root=foo)
chkpt.write("/tmp/foo")

del foo, chkpt
fresh_foo = Foo()
fresh_chkpt = tf.train.Checkpoint(root=fresh_foo)
assert fresh_foo.bar.numpy() == 0
fresh_chkpt.read("/tmp/foo")
assert fresh_foo.bar.numpy() == 2
print("Completed successfully")
```

During training, our training state will be made up of:

- the [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), including any trainable weights, the random state of any stochastic operations, and the [Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer) if provided in [compile](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile);
- the training data [Iterator](https://www.tensorflow.org/api_docs/python/tf/data/Iterator), including random state associated with any data augmentation or shuffle operations, or buffers for operations like [prefetch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch); and
- any other stateful [callbacks][callback], like [ReduceLROnPlateau][reduce-lr-on-plateau] or [EarlyStopping][early-stopping].

For convenience, we'll be using [BackupAndRestore][backup-and-restore] callback to manage when training state is saved or restored via a `Checkpoint`. Unfortunately, `BackupAndRestore` only stores the state of the `Model` - not the other aspects of our training state listed above. A simple work-around is to include the other elements in our `Model`'s training state. Since `Model`s are `Module`s, they automatically track dependencies assigned as attributes.

```python
model._train_iter = train_iter
model_callbacks = callbacks
# now `model`'s state will include that state of `train_iter` and `callback`
```

## Random Seeds and Weight Initialization

Probably the largest source of non-determinism - and the simplest to fix - is weight initialization. We can make this deterministic by calling [tf.random.set_seed](https://www.tensorflow.org/api_docs/python/tf/random/set_seed).

```python
tf.random.set_seed(seed)
```

Note this will only affect operations created _after_ this call that use the global seed, including [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) weight initialization and [Dataset.shuffle](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle). It will _not_ affect the [global generator state](https://www.tensorflow.org/api_docs/python/tf/random/get_global_generator). If our program uses the global generator, we should also set it's state.

```python
tf.random.get_global_generator().reset_from_seed(seed)
```

Alternatively, we can replace the global generator with a fresh one.

```python
tf.random.set_global_generator(tf.random.Generator.from_seed(seed))
```

The former will operations created by methods on the global generator, while the second will not (though may cause garbage collection related breakages if there are no refences to the old global generator). The result should be equivalent if called before any other `get_global_generator` calls.

## Data Pipelining

Most training data pipelines will have up to 3 sources of randomness:

1. random operations involved in data augmentations like possible image rotations and/or flips;
2. race conditions associated with parallel map functions for data loading and augmentation; and
3. dataset shuffling.

While researching this post I realized incorporating the first two in a deterministic and pre-emptible manner with tensorflow is distinctly non-trivial. To keep things simple I refactored that section into an entirely [separate article](../deterministic-tf-part-2).

In this post, we'll use a relatively straight-forward pipeline without augmentation that uses a `shuffle` and uniform `map` function using the [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) interface. While `Dataset`s can be saved in checkpoints, they won't contain all the state we want - in our instance, the shuffle buffer. Instead, we want to work with the [Iterator](https://www.tensorflow.org/api_docs/python/tf/data/Iterator) of an infinite dataset.

```python
dataset: tf.data.Dataset = load_dataset(split='train')
examples_per_epoch = len(dataset)
dataset = dataset.repeat()
dataset = dataset.shuffle(shuffle_buffer, seed=0)
dataset = dataset.batch(batch_size)
dataset = dataset.map(
    map_func,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    deterministic=True,  # not strictly necessary - this will be the default behaviour
)
steps_per_epoch = examples_per_epoch // batch_size  # ignore final fractional batch
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_iter = iter(dataset)  # this can be saved in checkpoints to track buffer state
```

Note that we apply `repeat` _before_ `shuffle`. This has the following consequences:

- we have one persistent shuffle buffer, meaning we won't need to refill it from scratch each epoch;
- examples will bleed from one epoch to the next - i.e. every epoch will have slightly different examples; and
- our dataset has infinite length.

## Floating Point Determinism

There was a time when GPU operations were mostly non-deterministic due to race conditions in floating point operations. This is still the default case for many operations, but most can now be made deterministic by setting the `TF_DETERMINISTIC_OPS` environment variable.

```bash
export TF_DETERMINISTIC_OPS=1
```

Alternatively, we can set it inside python.

```python
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
```

See nvidia's [determinism repository](https://github.com/NVIDIA/framework-determinism) for full details. Note there is not universal coverage for deterministic implementations - exceptions include `tf.gather` gradients, `tf.math.segment_*` operations and sparse-dense matrix multiplications (though the repository discusses ongoing work to resolve these).

## Operations with Random State

Some operations like [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) are intended to be stochastic. Unfortunately, despite the [official guide for random number generation](https://www.tensorflow.org/guide/random_numbers) discouraging their use, most implementations in tensorflow use base `tf.random` operations, rather than `tf.random.stateless_*` variants or [Generator](https://www.tensorflow.org/api_docs/python/tf/random/Generator) methods. Hopefully this will change in subsequent releases, but for the moment we can re-implement those necessary for our networks. A simple dropout implementation is given below.

```python
# We register it so we don't serialize then deserialize a `tf.keras.layers.Dropout`
@tf.keras.utils.register_keras_serializable("PreEmptible")
class Dropout(tf.keras.layers.Layer):
    def __init__(self, rate: float, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._rate = rate
        self._seed = seed
        self._rng = None

    def get_config(self):
        config = super().get_config()
        config.update(dict(rate=self.rate, seed=self.seed))
        return config

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    def build(self, input_shape):
        if self.built:
            return
        assert self._rng is None
        if self.seed is None:
            self._rng = tf.random.get_global_generator().split(1)[0]
        else:
            self._rng = tf.random.Generator.from_seed(self.seed)
        super().build(input_shape)

    def _apply_training(self, inputs):
        mask = self._rng.uniform(tf.shape(inputs)) > self.rate
        return tf.where(mask, inputs / (1 - self.rate), tf.zeros_like(inputs))

    @tf.function
    def call(  # pylint: disable=arguments-differ
        self, inputs, training: Optional[bool] = None
    ):
        assert self._rng is not None
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            return self._apply_training(inputs)
        return tf.identity(inputs)
```

Because both `Layer`s and `Generator`s are `Module`s, by assigning our `Generator` as an attribute of our `Layer`, the `Generator` state will be saved anywhere our `Layer` is, including when it's part of a `Model`.

## Callbacks

[Callback][callback]s provide a flexible interface for users to inject custom behaviour into a training loop (e.g. `Model.fit`). Most implementations in `tf.keras.callbacks` are responsible for logging, saving, or otherwise providing user feedback on the training process. However, a couple directly influence the training process and maintain their own state based on performance across multiple epochs: [ReduceLROnPlateau][reduce-lr-on-plateau] and [EarlyStopping][early-stopping].

There are three things we need to do to ensure their state is included with our model's state:

- implement wrappers that extend `Module`;
- change stateful attributes to non-trainable[Variable](https://www.tensorflow.org/api_docs/python/tf/Variable)s rather than python primitives; and
- assign them as attributes to the `Model`.

In the following example, we demonstrate how we can wrap `ReduceLROnPlateau` such that it can be used in a pre-emptible training process.

```python
def variable_property(name: str, dtype: tf.DType, doc: Optional[str] = None, **kwargs):
    """
    Get a property that wraps `tf.Variable` assignment.

    Useful for augmenting a base class to save values in `tf.Variable`s rather than
    as attributes.
    """
    attr_name = f"_variable_{name}"

    def getx(self):
        return tf.keras.backend.get_value(getattr(self, attr_name))

    @tf.Module.with_name_scope
    def setx(self, value):
        variable = getattr(self, attr_name, None)
        if variable is None:
            variable = tf.Variable(value, dtype=dtype, name=name, **kwargs)
            setattr(self, attr_name, variable)
        else:
            variable.assign(value)

    def delx(self):
        delattr(self, attr_name)

    return property(getx, setx, delx, doc)


class CallbackModule(tf.Module):
    def set_model(self, model: tf.keras.Model):
        # pylint: disable=protected-access
        old_model = getattr(self, "model", None)
        if old_model is not None:
            del old_model._callbacks[self.name]
        if not hasattr(model, "_callbacks"):
            model._callbacks = {}

        callbacks = model._callbacks
        # pylint: enable=protected-access
        assert self.name not in callbacks
        callbacks[self.name] = self
        self.model = model  # pylint: disable=attribute-defined-outside-init


class ReduceLROnPlateau(base.ReduceLROnPlateau, CallbackModule):
    def __init__(self, **kwargs):
        CallbackModule.__init__(self, name=None)
        self._supports_tf_logs = True
        self._config = dict(kwargs)
        base.ReduceLROnPlateau.__init__(self, **self._config)

    best = variable_property("best", tf.float32, trainable=False)
    wait = variable_property("wait", tf.int64, trainable=False)
    cooldown_counter = variable_property("cooldown_counter", tf.int64, trainable=False)
```

## Custom Fit

While the intent of this post was to create an implementation that used everyone's favourite [Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit), I wasn't able to find a way. I suspect the issue is related to how keras iterates over the data, but there also seems to be some issues with the optimizer as well (deterministic results are acheivable with `Model.fit` if using `SGD` optimizer and without a `shuffle` transform in the data pipeline). Having said that, writing a custom `fit` implementation with the same interface isn't too onerous.

```python
def as_infinite_iterator(
    dataset: tf.data.Dataset, steps_per_epoch: Optional[int] = None
) -> Tuple[tf.data.Iterator, int]:
    """
    Get an iterator for an infinite dataset and steps_per_epoch.

    Args:
        dataset: possibly infinite dataset.
        steps_per_epoch: number of steps per epoch if `dataset` has infinite
            cardinality, otherwise `None` or `dataset`'s cardinality.

    Returns:
        iterator: tf.data.Iterator of possibly repeated `dataset`.
        steps_per_epoch: number of elements in iterator considered one epoch.

    Raises:
        ValueError is dataset has finite cardinality inconsistent with steps_per_epoch.
    """
    cardinality = tf.keras.backend.get_value(dataset.cardinality())
    if steps_per_epoch is None:
        steps_per_epoch = cardinality
        if cardinality == tf.data.INFINITE_CARDINALITY:
            raise ValueError(
                "steps_per_epoch must be provided if dataset has infinite "
                "cardinality"
            )
        dataset = dataset.repeat()
    elif cardinality != tf.data.INFINITE_CARDINALITY:
        assert cardinality == steps_per_epoch
        dataset = dataset.repeat()
    return iter(dataset), steps_per_epoch


def fit_custom(
    model: tf.keras.Model,
    x: tf.data.Dataset,
    epochs: int = 1,
    initial_epoch: int = 0,
    validation_freq: int = 1,
    validation_data: Optional[tf.data.Dataset] = None,
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
    callbacks: Iterable[tf.keras.callbacks.Callback]=(),
    verbose: bool=True,
) -> tf.keras.callbacks.History:
    """Custom fit implementation. See `tf.keras.Model.fit` for more info."""
    train_func = model.make_train_function()
    train_iter, steps_per_epoch = as_infinite_iterator(x, steps_per_epoch)
    model._train_iter = train_iter

    cb = tf.keras.callbacks.CallbackList(
        callbacks=callbacks, add_history=True, add_progbar=verbose, model=model
    )
    cb.set_params(dict(epochs=epochs, verbose=int(verbose), steps=steps_per_epoch))

    cb.on_train_begin()

    initial_epoch = model._maybe_load_initial_epoch_from_ckpt(initial_epoch)

    model.stop_training = False
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        cb.on_epoch_begin(epoch)

        logs = None
        for step in range(steps_per_epoch):
            cb.on_train_batch_begin(step)
            logs = train_func(train_iter)
            cb.on_train_batch_end(step, logs)
            if model.stop_training:
                break
        assert logs is not None
        epoch_logs = logs
        if validation_data is not None and model._should_eval(epoch, validation_freq):
            logs = model.evaluate(
                validation_data,
                steps=validation_steps,
                callbacks=callbacks,
                return_dict=True,
            )
            epoch_logs.update({"val_" + name: val for name, val in logs.items()})
        cb.on_epoch_end(epoch, epoch_logs)
        training_logs = epoch_logs
        if model.stop_training:
            break
    cb.on_train_end(logs=training_logs)
    del model._train_iter
    return model.history

```

## Complete Example

The above functionality is all implemented in my [kblocks](https://github.com/jackd/kblocks.git) repository. You can see a complete example (including data augmentation) [here](https://github.com/tensorflow/tensorflow/jackd/kblocks/examples/fit.py).

## Conclusion

As we've seen, there's a lot more to deterministic and pre-emptible training than just setting the random seed and adding a [BackupAndRestore][backup-and-restore].

<!-- _Think I'm wrong? Found a typo or bug? Have a question or just think something needs further explanation? Let me know in the comments._ -->

[callback]: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
[reduce-lr-on-plateau]: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
[early-stopping]: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
[backup-and-restore]: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/experimental/BackupAndRestore
