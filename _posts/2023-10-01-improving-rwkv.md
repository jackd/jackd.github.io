---
title: "Faster LLMs: Improving RWKV with Parallel Cumulative Sums"
date: 2023-10-01 11:00:01 +1000
image: https://github.com/jackd/keras-rwkv/blob/master/images/benchmark-032.png?raw=true
categories: [Tensorflow, pytorch, jax, keras, LLM]
tags: [performance, benchmarks]
math: true
---

<script>
window.MathJax = {
    tex: {
      tags: 'ams',
      inlineMath: [['$', '$'], ['\\(', '\\)']]
    }
};
</script>

Large language models are all the craze right now. I was keen to learn about [keras-nlp](https://keras.io/keras_nlp/) - keras' natural language processing framework - and recent methods, so I decided to implement [RWKV](https://arxiv.org/abs/2305.13048), a popular model originally implemented in pytorch that's fostered a surprisingly large ecosystem of tools and use cases. While doing so certainly gave me a good understanding of `keras-nlp` and the `RWKV` model, it also led to an implementation with potential to be much faster than the original.

## TL;DR

- The `WKV` implementation critical to RWKV models can be implemented as a cumulative sum;
- naive implementation of this cumulative sum leads to numerical issues, but these can be resolved with relatively standard tools;
- the resulting implementation is parallelizable using _associative scan_ / _prefix sum_ implementations that are available in most deep learning frameworks and accelerator libraries (including triton, cuda, tensorflow and jax);
- microbenchmarks and training times with this implementation show promise, though further experiments with more compute are required to understand if these benefits are present in scaled up environments; and
- the code is available at [github.com/jackd/keras-rwkv](https://github.com/jackd/keras-rwkv)

## RWKV

First, some background. RWKV (pronounced "RuWaKVuh") is named after the four key quantities ($R$, $W$, $K$ and $V$) used in the self-attention mechanism. This post focuses on the WKV part, which is given in the [original paper](https://arxiv.org/abs/2305.13048) as

$
z^\prime_t = \frac{\sum_{i=1}^{t-1}\exp(-(t - 1 - i)w + k_i) v_i + \exp(u + k_t) v_t}{\sum_{i=1}^{t-1}\exp(-(t - 1 - i)w + k_i) + \exp(u + k_t)}.
$

Multiplying top and bottom by $\exp((t - 1)w)$ yields

$
z^\prime_t = \frac{\sum_{i=1}^{t-1} \exp(k_i + i w)v_i + \exp(u - w + k_t + t_w)v_t}{\sum_{i=1}^{t-1} \exp(k_i + i w) + \exp(u - w + k_t + t_w)}.
$

If we let $\tilde{k}_n = k_n + n w$, this simplifies to

$
z^\prime_t = \frac{\sum_{i=1}^{t-1} \exp(\tilde{k}_i)v_i + \exp(u - w + \tilde{k}_t)v_t}{\sum_{i=1}^{t-1} \exp(\tilde{k}_i) + \exp(u - w + \tilde{k}_t)}.
$

This can be computed efficiently using a cumulative sum.

```python
import jax.numpy as jnp

def wkv_numerator(
    v, # [T, C]
    k, # [T, C]
    u, # [C]
    w, # [C]
):
    T, C = v.shape
    kt = k + jnp.arange(T, dtype=k.dtype)[:, None] * w
    accumulation = jnp.cumsum(jnp.exp(kt) * v, axis=0)
    offset = jnp.exp(u - w + kt) * v
    numer = accumulation[:-1] + offset[1:]
    return jnp.concatenate((v[:1], numer), axis=0)

def wkv(v, k, u, w):
    return wkv_numerator(v, k, u, w) / wkv_numerator(jnp.ones_like(v), k, u, w)
```

There are multiple benefits to this include:

- simplicity: no custom cuda kernels or hand-written backward passes; and
- parallelism: `cumsum` can be parallelized along the `T` dimension.

The major downside is that evaluating `exp(kt)` is numerically infeasible for long time sequences. To resolve this, we introduce an _exponentially weighted_ parameterization.

## Exponentially Weighted Parameterization

We define an exponentially weighted parameterization of a value $z$ as

$$
z = \exp(t) v,
$$

where we assume $t$ and $v$ are both $\mathcal{O}(1)$. Due to the exponential however, the scales of $z$ can vary dramatically. We can add two exponentially weighted values and return the exponentially weighted parameterization without explicitly evaluating either of them,

```python
import jax.numpy as jnp

def add(z1, z2):
    v1, t1 = z1
    v2, t2 = z2
    t_out = jnp.logaddexp(t1, t2)
    v_out = jnp.exp(t1 - t_out) * v1 + jnp.exp(t2 - t_out) * v2
    return v_out, t_out
```

## Exponentially Weighted WKV

To make out `wkv` implementation numerically stable, we simply replace the `cumsum` with a version that supports a custom `add` operation - `jax.lax.associative_scan`. Note the resulting exponentially weighted values have `t` values corresponding to the denominator in the original expression, so there's no need to compute a separate denominator.

```python
def wkv(v, k, w, u):
    sequence_length = k.shape[1]
    kt = k + jnp.arange(sequence_length)[:, None] * w
    v_acc, t_acc = jax.lax.assoociative_scan(add, (v, kt), axis=0)
    v_out, t_out = add((v_acc[:-1], t_acc[:-1]), (v[1:], u - w + kt))
    return jnp.concatenate((v[:1], v_out), axis=0)
```

Note that `associative_scan` (a.k.a. `prefix_sum`) is a fundamental operation in computer science that has been extensively studied. In particular, it is worth noting that work-efficient parallel implementations exist and are available in `cuda`, `jax`, `triton` (currently only nightly) and `tensorflow-probability`.

## Benchmark Results

So how does our implementation stack up against the custom CUDA kernel used in the original RWKV model? Well... it's difficult to say, because I've had to make do with thrashing my laptop which can't really support anything but the smallest implementations in highly unrealistic training scenarios. That said, the results we do have look promising.

For a small number of channels we are able to get significant speed-up seemingly constant computation time up to a very high number of time steps (it's probably $\mathcal{O}(\text{log}(T))$, but compared to $\mathcal{O}(T)$ that looks pretty constant). For extremely long sequences we see an uptick in computation time, probably due to core saturation. Below is a plot using an 32-dimensional embedding.

![Microbenchmark computation time with 32 dimensional embeddings](https://github.com/jackd/keras-rwkv/blob/master/images/benchmark-032.png?raw=true)

Admittedly standard embeddings are much higher dimension than this. If we increase it to 256, we see the uptick occur earlier in the jax implementation. I can't explain why the tensorflow implementation remains fast for so long.

![Microbenchmark computation time with 256 dimensional embeddings](https://github.com/jackd/keras-rwkv/blob/master/images/benchmark-256.png?raw=true)

So do these micro-benchmark improvements result in meaningful improvements in training speed? Well... again, it's hard to say for certain because my laptop wasn't designed to train large language models. I did prepare a _very_ dirty training script that runs the smallest model with a batch size of 2 however, and the results are again promising. The parallel jax implementation trained the fastest, with a 31% reduction in training time compared to the original CUDA implementation.

![Train step times.](https://github.com/jackd/keras-rwkv/blob/master/images/train-times.png?raw=true)

Having said all that, there are a few major disclaimers that should be made:

- as mentioned previously, these are highly unrealistic training scenarios on highly unrealistic hardware. If anyone wants to provide the necessary compute, I'd be very happy to run more realistic evaluations;
- all these timings are based on my own keras implementation. The latest version of keras is still very new and it wouldn't surprise me if tensorflow and jax optimizations have been prioritised over pytorch;
- I've never done much with pytorch. I attempted to use `torch.compile` for the above, but there were errors that needed suppressing; and
- Similarly, I've never done much with triton, and I'm not sure if I'm using `associative_scan` correctly in that context.

As such, treat the results presented above as a proof of concept and demonstration of potential, rather than indicative of performances on a realistic scale.

That's all for today. Check out the [repo](https://github.com/jackd/keras-rwkv) if you want to have a play around. I'll write another post about my experience with keras/keras-nlp soon, but until then, happy coding!
