# -*- coding: utf-8 -*-
"""Wavelet transform functions.
"""
import numpy as np
import pywt
from sigpy import backend, util

__all__ = ['fwt', 'iwt']


def get_wavelet_shape(shape, wave_name, axes, level):
    zshape = [((i + 1) // 2) * 2 for i in shape]

    tmp = pywt.wavedecn(
        np.zeros(zshape), wave_name, mode='zero', axes=axes, level=level)
    tmp, coeff_slices = pywt.coeffs_to_array(tmp, axes=axes)
    oshape = tmp.shape

    return oshape, coeff_slices


def fwt(input, wave_name='db4', axes=None, level=None):
    """Forward wavelet transform.

    Args:
        input (array): Input array.
        axes (None or tuple of int): Axes to perform wavelet transform.
        wave_name (str): Wavelet name.
        level (None or int): Number of wavelet levels.
    """

    print("In dev branch\n");
    device = backend.get_device(input)
    xp = device.xp

    wavdct = pywt.Wavelet(wave_name)
    dec_hi = xp.array(wavdct.dec_hi)
    dec_lo = xp.array(wavdct.dec_lo)

    if not axes:
        axes = [k for k in range(len(input.shape)) if input.shape[k] > 1]
    if not level:
        level = pywt.dwt_max_level(xp.min([input.shape[ax] for ax in axes]), dec_hi.size)
    assert(level > 0)

    # Zero padding.
    pad_array = [(0, dec_hi.size - 1 + (input.shape[k] + dec_hi.size - 1) % 2) if k in axes else (0, 0) \
                 for k in range(len(input.shape))]
    x = xp.pad(input, pad_array, 'constant', constant_values=(0, 0))
    X = xp.fft.fftn(x, axes=axes)

    # Dictionary naming convention:
    #   L: Low pass filter decomposition.
    #   H: High pass filter decomposition.
    #   X: No filter applies.
    # For example, LXH implies:
    #   First dimension is low pass filtered.
    #   Second dimension is not modified.
    #   Third dimension is high pass filtered.
    # Numbers represent level of decomposition.
    keys = ['%d' % level]
    for ax in range(len(x.shape)):
        if ax not in axes:
            keys = [elm + 'X' for elm in keys]
        else:
            keys = [elm + 'L' for elm in keys] + [elm + 'H' for elm in keys]

    wav = {}
    f = xp.ones(x.shape).astype(xp.complex)
    for key in keys:
        subkeys = list(key)[1:]
        for k in range(len(subkeys)):
            subkey = subkeys[k]
            if subkey == 'L':
                lo = xp.zeros((x.shape[k],)).astype(xp.complex64)
                lo[:dec_lo.size] = dec_lo
                lo = xp.reshape(xp.fft.fftn(lo, axes=(0,)), \
                        [lo.size if k == s else 1 for s in range(len(x.shape))])
                f = f * lo
            elif subkey == 'H':
                hi = xp.zeros((x.shape[k],)).astype(xp.complex64)
                hi[:dec_hi.size] = dec_hi
                hi = xp.reshape(xp.fft.fftn(hi, axes=(0,)), \
                        [hi.size if k == s else 1 for s in range(len(x.shape))])
                f = f * hi
        y = xp.fft.ifftn(X * f, axes=axes)
        for ax in axes:
            y = xp.take(y, [t * 2 for t in range(0, y.shape[ax]//2)], axis=ax)
        f.fill(1)
        wav[key] = y

    if (level > 1):
        approx_key = [elm for elm in wav.keys() if 'H' not in elm].pop()
        approx_val = wav[approx_key]
        next_level = fwt(wav[approx_key], wave_name=wave_name, axes=axes, level=level-1)
        wav.update(next_level)
        wav.pop(approx_key)

    return wav


def iwt(input, oshape, coeff_slices, wave_name='db4', axes=None, level=None):
    """Inverse wavelet transform.

    Args:
        input (array): Input array.
        oshape (tuple of ints): Output shape.
        coeff_slices (list of slice): Slices to split coefficients.
        axes (None or tuple of int): Axes to perform wavelet transform.
        wave_name (str): Wavelet name.
        level (None or int): Number of wavelet levels.
    """
    device = backend.get_device(input)
    input = backend.to_device(input, backend.cpu_device)

    input = pywt.array_to_coeffs(input, coeff_slices, output_format='wavedecn')
    output = pywt.waverecn(input, wave_name, mode='zero', axes=axes)
    output = util.resize(output, oshape)

    output = backend.to_device(output, device)
    return output
