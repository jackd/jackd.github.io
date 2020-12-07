---
title: "Deterministic Tensorflow Part 2: Data Augmentation"
date: 2020-12-06 11:00:01 +1000
image: /assets/img/posts/pre-emptible-venn.jpg
categories: [Tensorflow, Training]
tags: [deterministic, pre-emptible]
---

Data augmentation is commonly used to artificially inflate the size of training datasets and teach networks invariances to various transformations. For example, image classification networks often train better when their datasets are augmented with random rotations, lighting adjustments and random flips. This article focuses on methods of performing augmentation that is both _deterministic_ (the same each time a program is run) and _pre-emptible_ (able to be interrupted and resumed without affecting results). Deciding which augmentations to apply for any given model is beyond the scope of this article.

This is Part 2 of a 2-part series that looks at deterministic, pre-emptible tensorflow. [Part 1](../deterministic-tf-part-1) looks at other aspects of training keras models.

## Motivating Example

We'll focus on the following simple example.

```python
import tensorflow as tf


def map_func(x):
    noise = tf.random.uniform(())
    return tf.cast(x, tf.float32) + noise



length = 5
epochs = 2
base = tf.data.Dataset.range(length)
ds = base.map(map_func)

for _ in range(epochs):
    print(list(ds.as_numpy_iterator()))

# First run:
# [0.36385977, 1.3164903, 2.7754397, 3.7108712, 4.238324]
# [0.81800365, 1.971394, 2.1719813, 3.0710397, 4.3042865]
# Second run:
# [0.7445557, 1.1573758, 2.3454256, 3.5037904, 4.5108995]
# [0.19572377, 1.0291697, 2.9865825, 3.8925676, 4.386469]
```

For deterministic pipelines, we want augmentations to be different across epochs, but the same across different runs of the program. The above script satisfies the first criteria but fails the second.

The simplest way of removing sources of non-determinism is to set the random seed. Tensorflow makes this easy with [tf.random.set_seed](https://www.tensorflow.org/api_docs/python/tf/random/set_seed). However, the following example reveals some unintented consequences.

```python
tf.random.set_seed(0)
ds = base.map(map_func)

for _ in range(epochs):
    print(list(ds.as_numpy_iterator()))

# First run:
# [0.019757032, 1.5400312, 2.5166783, 3.4683528, 4.1485605]
# [0.019757032, 1.5400312, 2.5166783, 3.4683528, 4.1485605]
# Second run run:
# [0.019757032, 1.5400312, 2.5166783, 3.4683528, 4.1485605]
# [0.019757032, 1.5400312, 2.5166783, 3.4683528, 4.1485605]
```

Clearly we've gained deterministic behaviour, but we've broken our data augmentation. While this kind of determinism might be [desirable in some circumstances](https://github.com/tensorflow/tensorflow/issues/44195), it certainly is not what we want for data augmentation.

One simple work-around is to use a [repeat](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#repeat)ed dataset and managing epochs ourselves.

```python
tf.random.set_seed(0)
ds = base.map(map_func).repeat(2)

iterator = ds.as_numpy_iterator()
for _ in range(epochs):
    print([next(iterator) for _ in range(length)])
# [0.019757032, 1.5400312, 2.5166783, 3.4683528, 4.1485605]
# [0.27331388,  1.1708031, 2.8691258, 3.7368858, 4.750617]
```

The second involves managing random state explicitly using [tf.random.Generator](https://www.tensorflow.org/api_docs/python/tf/random/Generator)s. Note this is the approach [encouraged by tensorflow for random number generation](https://www.tensorflow.org/guide/random_numbers) moving forward.

```python
def rng_map_func(x):
    noise = rng.uniform(())
    return tf.cast(x, tf.float32) + noise


rng = tf.random.Generator.from_seed(0)
ds = base.map(rng_map_func)

for _ in range(num_repeats):
    print(list(ds.as_numpy_iterator()))
# [0.31179297, 1.18098,   2.7613525, 3.1380515, 4.0275183]
# [0.4607407,  1.2356606, 2.1758924, 3.786038,  4.549028]
```

Unfortunately, high-level augmentation ops (e.g. [tf.image.random_flip_left_right](https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right)) are currently still implemented in terms of [tf.random] ops, so you may have to re-implement these in terms of `tf.random.Generator`.

Of course, it's good practice to parallelize data augmentation - particularly if it operates on individual examples (as opposed to batches). [Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) has `num_parallel_calls` and `deterministic` arguments exactly for this reason. Unfortunately, setting `deterministic=True` [does not guarantee determinism](https://github.com/tensorflow/tensorflow/issues/44491) as shown in the following example.

```python
tf.random.set_seed(0)
ds = base.map(map_func, num_parallel_calls=4, deterministic=True)

for _ in range(epochs):
    print(list(ds.as_numpy_iterator()))

# First run:
# [0.14856052, 1.019757, 2.5400312, 3.4683528, 4.5166783]
# [0.019757032, 1.1485605, 2.5400312, 3.4683528, 4.5166783]
# Second run:
# [0.5400312, 1.019757, 2.4683528, 3.5166783, 4.1485605]
# [0.14856052, 1.019757, 2.5166783, 3.4683528, 4.5400314]
```

At first glance it seems like we've somehow gained variation in augmentation at the cost of determinism across runs. Closer inspection however reveals the variation from the `tf.random.uniform` - the fractional parts - is the same for each iteration and each run, just applied to different elements. This indicates the non-determinism is due to a race condition rather than different sequence of random states. Setting `determinism=True` [guarantees elements are returned in the order they are received, but says nothing about the order in which they are computed](https://github.com/tensorflow/tensorflow/issues/44491#issuecomment-733892901). For this reason, switching to a `tf.random.Generator` will do nothing to solve the problem.

To resolve this, we need tie a different random state to each example. We can do this using a [tf.data.experimental.RandomDataset](https://www.tensorflow.org/api_docs/python/tf/data/experimental/RandomDataset) and using [tf.random.stateless_uniform](https://www.tensorflow.org/api_docs/python/tf/random/stateless_uniform) inside our map function.

```python
def stateless_map_func(x, example_seed):
    noise = tf.random.stateless_uniform((), seed=example_seed)
    return tf.cast(x, tf.float32) + noise


seeds = tf.data.experimental.RandomDataset(seed=0).batch(2)
ds = tf.data.Dataset.zip((base.repeat(epochs), seeds)).map(
    stateless_map_func, num_parallel_calls=4
)
iterator = ds.as_numpy_iterator()
for _ in range(epochs):
    print([next(iterator) for _ in range(length)])

# [0.99265265, 1.9662285, 2.6942484, 3.1795664, 4.277122]
# [0.34727395, 1.9636483, 2.3396842, 3.0605106, 4.4146137]
```

We have to be careful with the order of our dataset transformations. If we were to `repeat` after [zip](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#zip)ed, the seeds from our `RandomDataset` would be the same for each epoch and we would lose the variation we want for data augmentation.

Take aways:

- If using `tf.random` operations (e.g. `tf.random.uniform`) use a `repeated` datasets.
- If using `tf.random` operations or methods from a `tf.random.Generator`, always `map` with `num_parallel_calls=1`.
- For parallel, deterministic augmentation, use `tf.random.stateless_*` operations in conjunction with a `tf.random.experimental.RandomDataset`.

## Saving and Restoring State

Pre-emptibility - the ability of a program to recover from a failure - is critical for long processes. Tensorflow has good state-saving capabilities provided by [Checkpoint](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)s. If we elect to manage state explicitly using a `tf.random.Generator`, then we simply save the `Generator` instance between epochs.

```python
path = "/tmp/rng-state"
ds = base.map(rng_map_func)
print(list(ds.as_numpy_iterator()))
# [0.31179297, 1.18098, 2.7613525, 3.1380515, 4.0275183]
chkpt = tf.train.Checkpoint(rng=rng)
chkpt.write(path)
print(list(ds.as_numpy_iterator()))  # different to above
# [0.4607407, 1.2356606, 2.1758924, 3.786038, 4.549028]

chkpt.read(path)
print(list(ds.as_numpy_iterator()))  # same as above
# [0.4607407, 1.2356606, 2.1758924, 3.786038, 4.549028]
```

Unfortunately, this won't work with pipelines involving other random transformations like [shuffle](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) or operations that maintain an internal buffer like [prefetch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch). For those, we need to save the [iterator](https://www.tensorflow.org/api_docs/python/tf/data/Iterator) rather than the `dataset`.

```python
random_ds = tf.data.experimental.RandomDataset(seed=0)
path = "/tmp/iterator-state"
ds = tf.data.Dataset.zip((base.repeat(), random_ds.batch(2)))
ds = ds.map(stateless_map_func, num_parallel_calls=4)
iterator = iter(ds)
print([iterator.next().numpy() for _ in range(length)])
# [0.99265265, 1.9662285, 2.6942484, 3.1795664, 4.277122]
chkpt = tf.train.Checkpoint(iterator=iterator)
chkpt.write(path)
print([iterator.next().numpy() for _ in range(length)])  # different to above
# [0.34727395, 1.9636483, 2.3396842, 3.0605106, 4.4146137]

chkpt.read(path)
print([iterator.next().numpy() for _ in range(length)])  # same as above
# [0.34727395, 1.9636483, 2.3396842, 3.0605106, 4.4146137]
```

Note here we used stateless operations along with a random dataset. If we wanted to use a `Generator` (and `map` with `num_parallel_calls=1`) we could - we would just have to include it in our checkpoint alongside the iterator.

## Decoupling Augmentation from RNG Implementation

If we're writing an augmentation function for an image classification pipeline, we want to focus on the ideas of augmentation - whether it is appropriate to flip horizontally and/or vertically, how much lighting variation we can get away with etc. We don't want this code made more complicated with ideas related to the random number generation itself such as managing seeds for stateless operations or passing around `Generator` instances.

To illustrate this, consider the problem of applying a transformation with random elements to an existing dataset. We might use the following in our data pipeline code.

```python
def apply_stateless_map(
        dataset: tf.data.Dataset, map_func: Callable, seed: int=0, **map_kwargs
) -> tf.data.Dataset:
    seeds_dataset = tf.data.experimental.RandomDataset(seed).batch(2)
    zipped = tf.data.Dataset.zip((seeds_dataset, dataset))
    return zipped.map.map(map_func, **map_kwargs)
```

An appropriate `map_func` might be defined as follows.

```python
def random_shift(element_seed, element):
    tf.shape(element)
    shift = tf.random.stateless_normal(shape, seed=element_seed)
    element = element + shift
    return element
```

But what if we wanted to add an additional random transformation? A scale perhaps? We would rather not have to go into `apply_map` and change the shape of the seed. We could instead use `tf.random.stateless_split` to create a fresh seed from the existing one.

```python
def random_shift_and_scale(element_seed, element):
    shape = tf.shape(element)
    shift = tf.random.stateless_normal(shape, seed=seed)
    element = element + shift
    # compute a fresh seed
    seed = tf.squeeze(tf.random.experimental.stateless_split(seed, 1), axis=0)
    scale = tf.random.stateless_normal(shape, stddev=0.1, mean=1.0, seed=seed)
    element = element * scale
    return element
```

This is better, but this `random_shift_and_scale` implementation depends on us using it with an implementation like `apply_stateless_map` above. What about if we wanted to switch to a generator-based pipeline implementation?

To resolve this, I created the small [tfrng](https://github.com/jackd/tfrng) module to abstract away the random number generation (RNG) implementation details from the augmentation code, specifying it instead in the data pipelining stage.

A full guide is beyond the scope of this post, but the above functionality can be achieved with the code below. If that peaks your interest, check out this [more complete example](https://github.com/jackd/tfrng/blob/master/examples)

```python
def random_shift_and_scale(element):
    shape = tf.shape(element)
    shift = tfrng.normal(shape)
    scale = tfrng.normal(shape, stddev=0.1, mean=1.0)
    return (element + shift) * scale


base = tf.data.Dataset.range(4, output_type=tf.float32)
dataset = base.apply(
    tfrng.data.stateless_map(random_shift_and_scale, seed=0, num_parallel_calls=4)
)
print(list(dataset.as_numpy_iterator()))
# [-0.097278975, 0.8430341, 1.5062429, 1.0853299]

```

## Putting it all together

Of course, it wouldn't be a machine learning post without an obligatory MNIST example, so let's put everything we've learned together to make a deterministic, pre-emptible and performant training data pipeline with data augmentation and shuffling that varies across epochs.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def map_func(batch_seed, element, noise_scale=0.1):
    images, labels = element
    images = tf.cast(images, tf.float32) / 255
    noise = tf.random.stateless_uniform(
        tf.shape(images), seed=batch_seed, maxval=noise_scale
    )
    images = (images + noise) / (1 + noise_scale)
    return images, labels


tf.random.set_seed(0)  # influences shuffle
base = tfds.load(
    "mnist",
    split="train",
    as_supervised=True,
    shuffle_files=True,  # MNIST only has 1 file, but this will make it work for others
    read_config=tfds.ReadConfig(
        shuffle_seed=0,  # dataset will be non-deterministic if we don't provide a seed
        skip_prefetch=True,  # We'll prefetch batched elements later
    ),
)
dataset = base.repeat()
dataset = dataset.shuffle(
    1024,
    seed=0,
    reshuffle_each_iteration=False,  # note this is being applied to an infinite dataset
)
dataset = dataset.batch(128)

seeds = tf.data.experimental.RandomDataset(seed=0).batch(2)
dataset = tf.data.Dataset.zip((seeds, dataset))
dataset = dataset.map(map_func, num_parallel_calls=AUTOTUNE, deterministic=True)

dataset = dataset.prefetch(AUTOTUNE)

iter0 = [el[0].numpy() for el in dataset.take(5)]
iter1 = [el[0].numpy() for el in dataset.take(5)]

for i0, i1 in zip(iter0, iter1):
    assert (i0 == i1).all()
print("Consistent!")

# verify same between runs
print([i.sum() for i in iter0])
# [16272.569, 16938.98, 16157.573, 16848.334, 16400.393]
```

One thing to note is we'll get slight bleeding of examples from one epoch to the next thanks to the number of examples not being divisible by the batch size, and the shuffle buffer taking examples from the new epoch before the old epoch is done (you can see this if you replace the base dataset with `base = tf.data.Dataset.range(10, output_type=tf.float32)`). In most cases this won't be an issue, but if it were we could reorder our transformations to `shuffle`, `batch`, `repeat`, `zip`, `map`, `pretch`. This may be slightly less performant since the shuffle buffer will have to be filled from scratch each epoch.

Things get trickier if we need to apply the `map` before the `batch`, but I'll leave that as an exercise (hint: check out [flat_map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flat_map)).

## Conclusion

As we've seen, there's a lot more to deterministic data augmentation than just setting seeds. Having said that, hopefully this article has demonstrated that tnesorflow provides all the tools to write deterministic, pre-emptible, performant and maintainable pipelines.

_Think I'm wrong? Found a typo or bug? Have a question or just think something needs further explanation? Let me know in the comments._
