
"""Position encodings and utilities."""

import abc
import functools

import numpy as np


def generate_fourier_features(
    pos, num_bands, max_resolution=(224, 244),
    concat_pos=True, sine_only=False):
  """Generate a Fourier frequency position encoding with linear spacing.

  Args:
    pos: The position of n points in d dimensional space.
      A jnp array of shape [n, d].
    num_bands: The number of bands (K) to use.
    max_resolution: The maximum resolution (i.e. the number of pixels per dim).
      A tuple representing resolution for each dimension
    concat_pos: Concatenate the input position encoding to the Fourier features?
    sine_only: Whether to use a single phase (sin) or two (sin/cos) for each
      frequency band.
  Returns:
    embedding: A 1D jnp array of shape [n, n_channels]. If concat_pos is True
      and sine_only is False, output dimensions are ordered as:
        [dim_1, dim_2, ..., dim_d,
         sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
         sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
         cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
         cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
       where dim_i is pos[:, i] and f_k is the kth frequency band.
  """
  min_freq = 1.0
  # Nyquist frequency at the target resolution:

  freq_bands = np.stack([
      np.linspace(min_freq, res / 2, num=num_bands, endpoint=True)
      for res in max_resolution], axis=0)

  # Get frequency bands for each spatial dimension.
  # Output is size [n, d * num_bands]

  per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
  per_pos_features = np.reshape(per_pos_features,
                                 [-1, np.prod(per_pos_features.shape[1:])])

  if sine_only:
    # Output is size [n, d * num_bands]
    per_pos_features = np.sin(np.pi * (per_pos_features))
  else:
    # Output is size [n, 2 * d * num_bands]
    per_pos_features = np.concatenate(
        [np.sin(np.pi * per_pos_features),
         np.cos(np.pi * per_pos_features)], axis=-1)
  # Concatenate the raw input positions.
  if concat_pos:
    # Adds d bands to the encoding.
    per_pos_features = np.concatenate([pos, per_pos_features], axis=-1)
  return per_pos_features

