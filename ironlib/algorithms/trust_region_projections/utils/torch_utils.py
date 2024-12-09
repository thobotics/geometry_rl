import logging
import math
import os
from typing import Any, Optional
import collections

import numpy as np
import torch as ch
from torch.distributions import constraints


def sqrtm_newton(x: ch.Tensor, **kwargs: Any):
    """
    From: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    License: MIT

    Compute the Sqrt of a matrix based on Newton-Schulz algorithm
    """
    num_iters = kwargs.get("num_iters") or 10

    batch_size = x.shape[0]
    dim = x.shape[-1]
    dtype = x.dtype

    normA = x.pow(2).sum(dim=1).sum(dim=1).sqrt()
    Y = x / normA.view(batch_size, 1, 1).expand_as(x)
    I = 3.0 * ch.eye(dim, dtype=dtype)
    Z = ch.eye(dim, dtype=dtype)
    for i in range(num_iters):
        T = 0.5 * (I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    sA = Y * normA.sqrt().view(batch_size, 1, 1).expand_as(x)
    return sA


def sqrtm(x: ch.Tensor):
    """
    Compute the Sqrt of a matrix based on eigen decomposition. Assumes the matrix is symmetric PSD.

    Args:
        x: data

    Returns:
        matrix sqrt of x
    """
    eigvals, eigvecs = x.symeig(eigenvectors=True, upper=False)
    return eigvecs @ (ch.sqrt(eigvals)).diag_embed(0, -2, -1) @ eigvecs.permute(0, 2, 1)


def torch_batched_trace(x) -> ch.Tensor:
    """
    Compute trace in n,m of batched matrix
    Args:
        x: matrix with shape [a,...l, n, m]

    Returns: trace with shape [a,...l]

    """
    return ch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def torch_batched_trace_square(x):
    """
    Compute trace in n,m of squared batched matrix XX^{T}
    Args:
        x: matrix with shape [a,..., l, n, m]

    Returns: trace with shape [a,...l]

    """
    n = x.size(-1)
    m = x.size(-2)
    flat_trace = x.reshape(-1, m * n).square().sum(-1)
    return flat_trace.reshape(x.shape[:-2])


def tensorize(x, cpu=True, dtype=ch.float32):
    return cpu_tensorize(x, dtype) if cpu else gpu_tensorize(x, dtype)


def gpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cuda tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        gpu tensor version of x
    """
    return cpu_tensorize(x, dtype).cuda()


def cpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cpu tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        cpu tensor version of x
    """
    dtype = dtype if dtype else x.dtype
    x = ch.as_tensor(x, dtype=dtype, device=ch.device("cpu"))
    return x


def to_gpu(x):
    """
    Utility function for turning tensors into gpu tensors
    Args:
        x: data

    Returns:
        gpu tensor version of x
    """
    return x.cuda()


def get_numpy(x):
    """
    Convert torch tensor to numpy
    Args:
        x: torch.Tensor

    Returns:
        numpy tensor version of x

    """
    return x.cpu().detach().numpy()


def flatten_batch(x):
    """
        flatten axes 0 and 1
    Args:
        x: tensor to flatten

    Returns:
        flattend tensor version of x
    """

    s = x.shape
    return x.contiguous().view([s[0] * s[1], *s[2:]])


def select_batch(index, *args) -> list:
    """
    For each argument select the value at index.
    Args:
        index: index of values to select
        *args: data

    Returns:
        list of indexed value
    """
    return [v[index] for v in args]


def generate_minibatches(n, n_minibatches):
    """
    Generate n_minibatches sets of indices for N data points.
    Args:
        n: total number of data points
        n_minibatches: how many minibatches to generate

    Returns:
        np.ndarray of minibatched indices
    """
    state_indices = np.arange(n)
    np.random.shuffle(state_indices)
    return np.array_split(state_indices, n_minibatches)


def fill_triangular(x, upper=False):
    """
    From: https://github.com/tensorflow/probability/blob/c833ee5cd9f60f3257366b25447b9e50210b0590/tensorflow_probability/python/math/linalg.py#L787
    License: Apache-2.0

    Creates a (batch of) triangular matrix from a vector of inputs.

    Created matrix can be lower- or upper-triangular. (It is more efficient to
    create the matrix as upper or lower, rather than transpose.)

    Triangular matrix elements are filled in a clockwise spiral. See example,
    below.

    If `x.shape` is `[b1, b2, ..., bB, d]` then the output shape is
    `[b1, b2, ..., bB, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
    `n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.

    Example:

    ```python
    fill_triangular([1, 2, 3, 4, 5, 6])
    # ==> [[4, 0, 0],
    #      [6, 5, 0],
    #      [3, 2, 1]]

    fill_triangular([1, 2, 3, 4, 5, 6], upper=True)
    # ==> [[1, 2, 3],
    #      [0, 5, 6],
    #      [0, 0, 4]]
    ```

    The key trick is to create an upper triangular matrix by concatenating `x`
    and a tail of itself, then reshaping.

    Suppose that we are filling the upper triangle of an `n`-by-`n` matrix `M`
    from a vector `x`. The matrix `M` contains n**2 entries total. The vector `x`
    contains `n * (n+1) / 2` entries. For concreteness, we'll consider `n = 5`
    (so `x` has `15` entries and `M` has `25`). We'll concatenate `x` and `x` with
    the first (`n = 5`) elements removed and reversed:

    ```python
    x = np.arange(15) + 1
    xc = np.concatenate([x, x[5:][::-1]])
    # ==> array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13,
    #            12, 11, 10, 9, 8, 7, 6])

    # (We add one to the arange result to disambiguate the zeros below the
    # diagonal of our upper-triangular matrix from the first entry in `x`.)

    # Now, when reshapedlay this out as a matrix:
    y = np.reshape(xc, [5, 5])
    # ==> array([[ 1,  2,  3,  4,  5],
    #            [ 6,  7,  8,  9, 10],
    #            [11, 12, 13, 14, 15],
    #            [15, 14, 13, 12, 11],
    #            [10,  9,  8,  7,  6]])

    # Finally, zero the elements below the diagonal:
    y = np.triu(y, k=0)
    # ==> array([[ 1,  2,  3,  4,  5],
    #            [ 0,  7,  8,  9, 10],
    #            [ 0,  0, 13, 14, 15],
    #            [ 0,  0,  0, 12, 11],
    #            [ 0,  0,  0,  0,  6]])
    ```

    From this example we see that the resuting matrix is upper-triangular, and
    contains all the entries of x, as desired. The rest is details:

    - If `n` is even, `x` doesn't exactly fill an even number of rows (it fills
      `n / 2` rows and half of an additional row), but the whole scheme still
      works.
    - If we want a lower triangular matrix instead of an upper triangular,
      we remove the first `n` elements from `x` rather than from the reversed
      `x`.

    For additional comparisons, a pure numpy version of this function can be found
    in `distribution_util_test.py`, function `_fill_triangular`.

    Args:
      x: `Tensor` representing lower (or upper) triangular elements.
      upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).

    Returns:
      tril: `Tensor` with lower (or upper) triangular elements filled from `x`.

    Raises:
      ValueError: if `x` cannot be mapped to a triangular matrix.
    """

    m = np.int32(x.shape[-1])
    # Formula derived by solving for n: m = n(n+1)/2.
    n = np.sqrt(0.25 + 2.0 * m) - 0.5
    if n != np.floor(n):
        raise ValueError(
            "Input right-most shape ({}) does not "
            "correspond to a triangular matrix.".format(m)
        )
    n = np.int32(n)
    new_shape = x.shape[:-1] + (n, n)

    ndims = len(x.shape)
    if upper:
        x_list = [x, ch.flip(x[..., n:], dims=[ndims - 1])]
    else:
        x_list = [x[..., n:], ch.flip(x, dims=[ndims - 1])]

    x = ch.cat(x_list, dim=-1).reshape(new_shape)
    x = ch.triu(x) if upper else ch.tril(x)
    return x


def fill_triangular_inverse(x, upper=False):
    """
    From: https://github.com/tensorflow/probability/blob/c833ee5cd9f60f3257366b25447b9e50210b0590/tensorflow_probability/python/math/linalg.py#L937
    License: Apache-2.0

    Creates a vector from a (batch of) triangular matrix.

    The vector is created from the lower-triangular or upper-triangular portion
    depending on the value of the parameter `upper`.

    If `x.shape` is `[b1, b2, ..., bB, n, n]` then the output shape is
    `[b1, b2, ..., bB, d]` where `d = n (n + 1) / 2`.

    Example:

    ```python
    fill_triangular_inverse(
      [[4, 0, 0],
       [6, 5, 0],
       [3, 2, 1]])

    # ==> [1, 2, 3, 4, 5, 6]

    fill_triangular_inverse(
      [[1, 2, 3],
       [0, 5, 6],
       [0, 0, 4]], upper=True)

    # ==> [1, 2, 3, 4, 5, 6]
    ```

    Args:
      x: `Tensor` representing lower (or upper) triangular elements.
      upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).

    Returns:
      flat_tril: (Batch of) vector-shaped `Tensor` representing vectorized lower
        (or upper) triangular elements from `x`.
    """

    n = np.int32(x.shape[-1])
    m = np.int32((n * (n + 1)) // 2)

    ndims = len(x.shape)
    if upper:
        initial_elements = x[..., 0, :]
        triangular_part = x[..., 1:, :]
    else:
        initial_elements = ch.flip(x[..., -1, :], dims=[ndims - 2])
        triangular_part = x[..., :-1, :]

    rotated_triangular_portion = ch.flip(
        ch.flip(triangular_part, dims=[ndims - 1]), dims=[ndims - 2]
    )
    consolidated_matrix = triangular_part + rotated_triangular_portion

    end_sequence = consolidated_matrix.reshape(x.shape[:-2] + (n * (n - 1),))

    y = ch.cat([initial_elements, end_sequence[..., : m - n]], dim=-1)
    return y


def diag_bijector(f: callable, x):
    """
    Apply transformation f(x) on the diagonal of a batched matrix.
    Args:
        f: callable to apply to diagonal
        x: data

    Returns:
        transformed matrix x
    """
    return x.tril(-1) + f(x.diagonal(dim1=-2, dim2=-1)).diag_embed() + x.triu(1)


def inverse_softplus(x):
    """
    x = inverse_softplus(softplus(x))
    Args:
        x: data

    Returns:

    """
    return (x.exp() - 1.0).log()


def torch_atleast_2d(x: ch.Tensor, reverse=False):
    """
    Transforms torch tensor to a torch tensor with at least a 2D shape.
    Args:
        x: data
        reverse: For 1D input only -> if True: x[:, None] else: x[None, :]

    Returns:
        2D torch tensor or input, when already larger than 2D
    """
    if x.dim() == 0:
        result = x.reshape([1, 1])
    elif x.dim() == 1:
        result = x[:, None] if reverse else x[None, :]
    else:
        result = x
    return result


def torch_atleast_3d(x: ch.Tensor, reverse=False):
    """
    Transforms torch tensor to a torch tensor with at least a 3D shape.
    Args:
        x: data
        reverse: For 1D input only -> if True: x[:, None] else: x[None, :]

    Returns:
        2D torch tensor or input, when already larger than 2D
    """
    if x.dim() == 0:
        result = x.reshape(1, 1, 1)
    elif x.dim() == 1:
        result = x[:, None, None] if reverse else x[None, None, :]
    elif x.dim() == 2:
        result = x[:, :, None] if reverse else x[None, :, :]
    else:
        result = x
    return result


def log1pexp(x: ch.Tensor):
    """
    Computes log(1 + exp(x)), which is e.g. used to add log probabilities.
    For details of the computation check here:
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    Args:
        x: probability in log-space
    """
    out = ch.zeros_like(x)
    mask_1 = x.le(-37)
    mask_2 = ~mask_1 * x.le(18)
    mask_3 = ~mask_1 * ~mask_2 * x.le(33.3)
    mask_4 = ~mask_1 * ~mask_2 * ~mask_3

    out[mask_1] = x[mask_1].exp()
    out[mask_2] = x[mask_2].exp().log1p()
    out[mask_3] = x[mask_3] + (-x[mask_3]).exp()
    out[mask_4] = x[mask_4]

    return out


def log1mexp(x: ch.Tensor):
    """
    Computes log(1 - exp(x)), which is e.g. used to subtract log probabilities.
    For details of the computation check here:
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    Args:
        x: probability in log-space
    """
    return ch.where(x <= math.log(2), ch.log(-ch.expm1(x)), ch.log1p(-ch.exp(x)))
    # dtype_max = ch.finfo(x.dtype).max
    # return 0 if x > dtype_max else (1 - x.exp()).log()


def logaddexp(input: ch.Tensor, other: ch.Tensor):
    """
    Add two log values. Naively this could be done as log(exp(input) + exp(other)).
    This version, however, is numerically more stable
    Args:
        input: first probability in log-space
        other: second probability in log-space
    """
    max_val = ch.maximum(input, other)
    return max_val + log1pexp(-ch.abs(input - other))


def logsubexp(input: ch.Tensor, other: ch.Tensor):
    """
    Subtract two log values. Naively this could be dones as log(exp(input) - exp(other)).
    This version, however, is numerically more stable
    Args:
        input: first probability in log-space
        other: second probability in log-space
    """
    assert (
        input < other
    ).all(), "Cannot subtract larger number from smaller one in log space, log is not defined"
    return input + log1mexp(other - input)


def get_stats_dict(d: dict):
    def get_stats(x, name):
        return {
            f"{name}_mean": x.mean(),
            f"{name}_std": x.std(),
            f"{name}_max": x.max(),
            f"{name}_min": x.min(),
        }

    out = collections.OrderedDict()
    for k, v in d.items():
        out.update(get_stats(v, k))
    return out


def try_set_gpu_device(params: dict):
    if not params["cpu"]:
        avail_gpus = ch.cuda.is_available()
        if avail_gpus:
            ch.cuda.set_device(avail_gpus)
            logging.info(f"Currently used GPUs: {ch.cuda.current_device()}")
        else:
            logging.warning("No GPU available, running on CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            params["cpu"] = True


def transform_to_2hot(
    scalar: ch.Tensor, min_value: float, max_value: float, num_bins: int
) -> ch.Tensor:
    """Transforms a scalar tensor to a 2 hot representation."""
    scalar = ch.clip(scalar, min_value, max_value)
    scalar_bin = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
    lower, upper = ch.floor(scalar_bin), ch.ceil(scalar_bin)
    lower_value = (lower / (num_bins - 1.0)) * (max_value - min_value) + min_value
    upper_value = (upper / (num_bins - 1.0)) * (max_value - min_value) + min_value
    p_lower = ((upper_value - scalar) / (upper_value - lower_value + 1e-5)).unsqueeze(
        -1
    )
    p_upper = 1 - p_lower
    lower_one_hot = one_hot(lower, num_bins, dtype=scalar.dtype) * p_lower
    upper_one_hot = one_hot(upper, num_bins, dtype=scalar.dtype) * p_upper
    return lower_one_hot + upper_one_hot


def one_hot(indices, num_classes, dtype=ch.float32):
    """Returns a one-hot version of indices.

    Args:
      indices: A tensor of indices.
      num_classes: Number of classes in the one-hot dimension.
      dtype: The dtype.

    Returns:
      The one-hot tensor. If indices' shape is [A, B, ...], shape is
      [A, B, ..., num_classes].
    """
    labels = ch.arange(num_classes)
    for _ in range(indices.ndim):
        labels = labels.unsqueeze(dim=0)
    return ch.as_tensor(indices[..., None] == labels, dtype=dtype)
