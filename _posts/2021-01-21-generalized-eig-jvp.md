---
title: "Generalized Eigenvalue Problem Derivatives"
date: 2021-01-21 11:00:01 +1000
image: /assets/img/posts/jax.png
categories: [Linear Algebra]
tags: [jax, autograd]
math: true
---

<!-- Enable equation numbering -->
<script>
window.MathJax = {
    tex: {
      tags: 'ams',
      inlineMath: [['$', '$'], ['\\(', '\\)']]
    }
};
</script>

The eigenvector problem is ubiquitous in many areas of mathematics, physics and computer science. I recently found myself needing the solution to the generalized eigenvalue problem and discovered an implementation doesn't exist in [jax][jax]. While wrapping [low-level cuda code](https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1) is mechanical enough, this doesn't help with one of the core features of jax: auto-differentiation.

A naive solution would be to re-write the algorithm that generates the decomposition. While this would get us a differentiable solution, it would suffer some major flaws. Most obviously:

* we'd have more code to maintain;
* it likely would not be as efficient as the CUDA implementation; and
* the automatically calculated derivatives would still be sub-optimal.

Instead, we're going to calculate derivatives using _implicit differentiation_. This is hardly a new idea, and there are numerous sources detailing results for the standard eigenvalue problem. That said, I was unable to find anything on the generalized problem. In this post, we'll start by formally defining the problem, before adapting the approach of [Boeddeker _et al._][boeddeker] to the generalized problem. In doing so, we identify what we believe to be a flaw in their solution, though identity a work-around for the self-adjoint case. A good understanding of the basics of linear algebra and calculus is assumed.

## Problem Description

<div>
\(\newcommand{\pd}[1]{\frac{\partial {#1}}{\partial \xi}}\)
\(\newcommand{\A}{\mathbf{A}}\)
\(\newcommand{\B}{\mathbf{B}}\)
\(\newcommand{\W}{\mathbf{W}}  % vectors\)
\(\newcommand{\V}{\mathbf{V}}  % values\)
\(\newcommand{\dA}{\pd{\A}}\)
\(\newcommand{\dB}{\pd{\B}}\)
\(\newcommand{\dW}{\pd{\W}}\)
\(\newcommand{\dWH}{\pd{\W^H}}\)
\(\newcommand{\dV}{\pd{\V}}\)
\(\newcommand{\Winv}{\W^{-1}}\)
\(\newcommand{\Binv}{\B^{-1}}\)
\(\newcommand{\Re}{\mathbb{R}e}\)
\(\newcommand{\Im}{\mathbb{I}m}\)
</div>

We consider the problem of computing partial derivatives of $\W$ and $\V$ given the solution to the generalized eigenvalue and all other relevant partial derivatives. We start with the definition of the solutions to the problem, i.e.

\begin{equation}
    \A\W = \B \W \V,
    \label{eqn:definition}
\end{equation}
where the columns of $\W$ are the eigenvectors, $\V$ is a diagonal matrix with eigenvalues on the diagonal and square matrices $\A$ and $\B$ define the problem. We limit discussion to the case where the eigenvalues are distinct.

We note there is a degree of freedom in the magnitude of the returned eigenvectors. We ignore this for the moment, but will come back to it.

## A Partial Solution to the General Problem

We begin by taking the partial deriative of Equation \ref{eqn:definition}:

\begin{equation}
    \dA \W + \A \dW = \dB \W \V + \B \dW \V + \B \W \dV
\end{equation}

Pre-multiplying by $\Winv \Binv$ yields
\begin{equation}
    \Winv \Binv \dA \W + \Winv \Binv \A \dW = \Winv \Binv \left[\B\dW + \dB \W \right] \V + \Winv \Binv \B \W \dV
\end{equation}

The second term on each side can be simplified:
\begin{equation}
    \Winv \Binv \dA \W + \V \Winv \dW = \Winv \Binv \left[\B \dW + \dB \W\right] \V + \dV
\end{equation}

Expanding and simplifying the first term on the right and shifting the second term on the left to the right yields
\begin{equation}
    \Winv \Binv \dA \W = \Winv \dW \V - \V \Winv \dW + \Winv \Binv \dB \W \V + \dV
\end{equation}

We can simplify the first two terms on the right by applying the identity $AD - DA = E \circ A$ for a diagonal matrix $D$, where $e_{ij} = d_{ii} - d_{jj}$ and $\circ$ denotes the Hadamard (element-wise) product.

\begin{equation}
    \Winv \Binv \dA \W = E \circ \Winv \dW + \Winv \Binv \dB \W \V + \dV
    \label{eqn:critical}
\end{equation}

By noting that $I \circ E = \mathbf{0}$, the derivatives of the eigenvalues $\V$ can be decoupled from those of $\W$ by taking the Hadamard product of the identity matrix with Equation \ref{eqn:critical}:
\begin{equation}
    \dV = I \circ \left[\Winv \Binv \left(\dA \W - \dB \W \V\right)\right].
\end{equation}

Extracting derivatives of $\W$ is trickier. To start with, we define matrix $F$ of the same size as $E$ with elements given by
<div>
\begin{equation}
    f_{ij} = \begin{cases}
      0 & i = j \\
      \frac{1}{e_ij} & i \neq j
   \end{cases}.
\end{equation}
</div>

By noting the $F \circ E = \mathbf{1} - I$, taking the Hadamard product of equation \ref{eqn:critical} with $F$ yields
\begin{equation}
    (\mathbf{1} - I)\circ \Winv \dW = F \circ \left[\Winv \Binv \left(\dA \W - \dB \W \V\right)\right].
    \label{eqn:troublesome}
\end{equation}

While everything on the right is straight-forward to compute, the Hadamard product on the left presents difficulties. [Giles][giles] proposes choosing scales of the eigenvectors such that the diagonal of $\Winv \dW$ are zero, though does not discuss how to compute such a scale. Note that contrary to the claims of [Boeddeker _et al._][boedekker], using normalized eigenvectors (or more generally, eigenvectors of fixed magnitude) does _not_ guarantee this property.

We make no effort to resolve this for the general case. We argue that use-cases of eigenvectors should be invariant to this scaling, and derivatives would likely be computed more easily with respect to these quantities instead. For the interested reader, a good discussion of the problem is given [here](https://re-ra.xyz/Gauge-Problem-in-Automatic-Differentiation/).

## Self-Adjoint Case

One special case where computing derivatives is with respect to eigenvectors is relatively straight forward is in the self-adjoint case, where $\A = \A^H$ and $\B = \B^H$. In this case, eigenvectors form an orthogonal basis. Equation \ref{eqn:troublesome} can then be computed by enforcing magnitudes such that the eigenvectors are orthogonal with respect to $\B$, i.e.
\begin{equation}
    \W^H \B \W = I \iff{\Winv = \W^H \B}.
    \label{eqn:orthonormal}
\end{equation}

This allows for some simplifications to Equation \ref{eqn:troublesome}:
\begin{equation}
    (\mathbf{1} - I)\circ \W^H \B \dW = F \circ \left[\W^H \left(\dA \W - \dB \W \V\right)\right].
    \label{eqn:less-troublesome}
\end{equation}

We still have the problem of the identity matrix in the Hadamard operator. We can distribute the product operator over addition ($[\mathbf{1} + I] \circ X = X + I \circ X$) but this still isn't in a form we can compute easily compute. Fortunately, the orthonormal constraint gives us a mechanism by which we can express the diagonal values of $\dW$ in terms of other known values.

To understand how, we begin by differentiating Equation \ref{eqn:orthonormal}
\begin{equation}
    \dWH \B \W + \W^H \dB \W + \W^H \B \dW = \mathbf{0}.
\end{equation}

Since the first and third term are Hermitian transposes of each other, their complex components will cancel out on the diagonal. This means we can expression the real part of our troublesome term in terms of less problematic components.
<div>
\begin{equation}
    I \circ \Re\left[\W^H \B \dW\right] = -\frac{1}{2} I \circ \W^H \dB \W.
    \label{eqn:identity-hadamard}
\end{equation}
</div>

Alas, this does not help us in evaluating the imaginary part. We do still have some freedom however: it can be shown if $\V, \W$ is a solution to the problem and satisfies the described orthogonality constraint, then $\V \mathbf{R}(\Theta)$ is also a solution, where $\mathbf{R}(\Theta)$ is a diagonal matrix with entries given by rotations in the complex plan, $R_{jj} = e^{i\theta_j}$. Exactly how we define $\theta_j$ in such a way that makes the remaining troublesome term computable is left to the reader (i.e. I've spent entirely too long trying and got nowhere).

## Real Symmetric Case

Like we did above, rather than overcome this dilemna we'll simply claim we were never interested in the solution to the complex problem anyway.

For real $\A$ and $\B$ the imaginary part is zero. Substituting Equation \ref{eqn:identity-hadamard} into Equation \ref{eqn:less-troublesome} and rearranging gives us our final expression for the eigenvector derivatives:
\begin{equation}
    \dW = \W \left(
      F \circ \left[\W^H\left(\dA \W - \dB \W \V\right)\right]
      - \frac{1}{2} I \circ \left[\W^H \dB \W\right]\right).
\end{equation}

## Jax Implementation

No amount of reviewing maths will ever make me trust a solution as much as numerical validation. We implemented the above in [jax][jax] (also available in [this gist](https://gist.github.com/jackd/99e012090a56637b8dd8bb037374900e)).

`eigh_impl.py`:

```python
"""Versions based on 4.60 and 4.63 of https://arxiv.org/pdf/1701.00392.pdf ."""
import jax
import jax.numpy as jnp
import numpy as np


def _T(x):
    return jnp.swapaxes(x, -1, -2)


def _H(x):
    return jnp.conj(_T(x))


def symmetrize(x):
    return (x + _H(x)) / 2


def standardize_angle(w, b):
    if jnp.isrealobj(w):
        return w * jnp.sign(w[0, :])
    else:
        # scipy does this: makes imag(b[0] @ w) = 1
        assert not jnp.isrealobj(b)
        bw = b[0] @ w
        factor = bw / jnp.abs(bw)
        w = w / factor[None, :]
        sign = jnp.sign(w.real[0])
        w = w * sign
        return w


@jax.custom_jvp  # jax.scipy.linalg.eigh doesn't support general problem i.e. b not None
def eigh(a, b):
    """
    Compute the solution to the symmetrized generalized eigenvalue problem.

    a_s @ w = b_s @ w @ np.diag(v)

    where a_s = (a + a.H) / 2, b_s = (b + b.H) / 2 are the symmetrized versions of the
    inputs and H is the Hermitian (conjugate transpose) operator.

    For self-adjoint inputs the solution should be consistent with `scipy.linalg.eigh`
    i.e.

    v, w = eigh(a, b)
    v_sp, w_sp = scipy.linalg.eigh(a, b)
    np.testing.assert_allclose(v, v_sp)
    np.testing.assert_allclose(w, standardize_angle(w_sp))

    Note this currently uses `jax.linalg.eig(jax.linalg.solve(b, a))`, which will be
    slow because there is no GPU implementation of `eig` and it's just a generally
    inefficient way of doing it. Future implementations should wrap cuda primitives.
    This implementation is provided primarily as a means to test `eigh_jvp_rule`.

    Args:
        a: [n, n] float self-adjoint matrix (i.e. conj(transpose(a)) == a)
        b: [n, n] float self-adjoint matrix (i.e. conj(transpose(b)) == b)

    Returns:
        v: eigenvalues of the generalized problem in ascending order.
        w: eigenvectors of the generalized problem, normalized such that
            w.H @ b @ w = I.
    """
    a = symmetrize(a)
    b = symmetrize(b)
    b_inv_a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(b), a)
    v, w = jax.jit(jax.numpy.linalg.eig, backend="cpu")(b_inv_a)
    v = v.real
    # with loops.Scope() as s:
    #     for _ in s.cond_range(jnp.isrealobj)
    if jnp.isrealobj(a) and jnp.isrealobj(b):
        w = w.real
    # reorder as ascending in w
    order = jnp.argsort(v)
    v = v.take(order, axis=0)
    w = w.take(order, axis=1)
    # renormalize so v.H @ b @ H == 1
    norm2 = jax.vmap(lambda wi: (wi.conj() @ b @ wi).real, in_axes=1)(w)
    norm = jnp.sqrt(norm2)
    w = w / norm
    w = standardize_angle(w, b)
    return v, w


@eigh.defjvp
def eigh_jvp_rule(primals, tangents):
    """
    Derivation based on Boedekker et al.

    https://arxiv.org/pdf/1701.00392.pdf

    Note diagonal entries of Winv dW/dt != 0 as they claim.
    """
    a, b = primals
    da, db = tangents
    if not all(jnp.isrealobj(x) for x in (a, b, da, db)):
        raise NotImplementedError("jvp only implemented for real inputs.")
    da = symmetrize(da)
    db = symmetrize(db)

    v, w = eigh(a, b)

    # compute only the diagonal entries
    dv = jax.vmap(
        lambda vi, wi: -wi.conj() @ db @ wi * vi + wi.conj() @ da @ wi, in_axes=(0, 1),
    )(v, w)

    dv = dv.real

    E = v[jnp.newaxis, :] - v[:, jnp.newaxis]

    # diagonal entries: compute as column then put into diagonals
    diags = jnp.diag(-0.5 * jax.vmap(lambda wi: wi.conj() @ db @ wi, in_axes=1)(w))
    # off-diagonals: there will be NANs on the diagonal, but these aren't used
    off_diags = jnp.reciprocal(E) * (_H(w) @ (da @ w - db @ w * v[jnp.newaxis, :]))

    dw = w @ jnp.where(jnp.eye(a.shape[0], dtype=np.bool), diags, off_diags)

    return (v, w), (dv, dw)
```

Dirty testing script:

```python
from eigh_impl import symmetrize, eigh, standardize_angle
import jax.numpy as jnp
import numpy as np

import jax.test_util as jtu
import scipy.linalg

jnp.set_printoptions(3)
rng = np.random.default_rng(0)

n = 5
is_complex = False


def make_spd(x):
    n = x.shape[0]
    return symmetrize(x) + n * jnp.eye(n)


def get_random_square(rng, size, is_complex=True):
    real = rng.uniform(size=size).astype(np.float32)
    if is_complex:
        return real + rng.uniform(size=size).astype(np.float32) * 1j
    return real


a = make_spd(get_random_square(rng, (n, n), is_complex=is_complex))
b = make_spd(get_random_square(rng, (n, n), is_complex=is_complex))

vals, vecs = eigh(a, b)
# ensure solution satisfies the problem
np.testing.assert_allclose(a @ vecs, b @ vecs @ jnp.diag(vals), atol=1e-5)
# ensure vectors are orthogonal w.r.t b
np.testing.assert_allclose(vecs.T.conj() @ b @ vecs, jnp.eye(n), atol=1e-5, rtol=1e-5)
# ensure eigenvalues are ascending
np.testing.assert_array_less(vals[:-1], vals[1:])
jtu.check_grads(eigh, (a, b), 2, modes=["fwd"])

# ensure values consistent with scipy
vals_sp, vecs_sp = scipy.linalg.eigh(a, b)
print("scipy")
print(vecs_sp)
print("this work")
print(vecs)
np.testing.assert_allclose(vals, vals_sp, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(vecs, standardize_angle(vecs_sp, b), rtol=1e-4, atol=1e-5)
print("success")
```

[boedekker]: https://arxiv.org/pdf/1701.00392.pdf
[giles]: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
[jax]: https://github.com/google/jax
