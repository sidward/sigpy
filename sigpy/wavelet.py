# -*- coding: utf-8 -*-
"""Wavelet transform functions.

Dimension dictionary convention:
    A: Approximation coefficients.
    D: Decimation coefficients.
    X: Nothing modified.

For example, "AXD" implies:
    First dimension contains the approximation coefficients.
    Second dimension is not modified.
    Third dimension contains the decimation coefficients.

    The prefix number represents level of decomposition.
"""
import numpy as np
import pywt
from sigpy import backend, util

import matplotlib.pyplot as plt

__all__ = ['fwt', 'iwt']


def get_wavelet_shape(shape, wave_name, axes, level):
    zshape = [((i + 1) // 2) * 2 for i in shape]

    tmp = pywt.wavedecn(
        np.zeros(zshape), wave_name, mode='zero', axes=axes, level=level)
    tmp, coeff_slices = pywt.coeffs_to_array(tmp, axes=axes)
    oshape = tmp.shape

    return oshape, coeff_slices


def apply_filter_along_axis(input, axes, filt_lo, filt_hi):
    """Apply filter along axis.

    Helper function to recursively apply wavelet filters along axes.

    Args:
        input (array): Input array.
        axes (tuple of int): Axes to perform wavelet transform.
        filt_lo (array): Wavelet coefficients for approximation coefficients.
        filt_hi (array): Wavelet coefficients for decimation coefficients.
    """
    assert type(axes) == tuple, "Second argument (axes) must be of type tuple."
    assert filt_lo.shape == filt_hi.shape, "Third argument (filt_lo) and fourth argument (filt_hi) must have the same shape"

    if (len(axes) == 0):
        return input

    # Loading sigpy.
    device = backend.get_device(input)
    xp = device.xp

    axis = axes[0]

    # Zero padding.
    pad_array = [(0, filt_hi.size - 1 + (input.shape[k] + filt_hi.size - 1) % 2) if k == axis else (0, 0) \
                    for k in range(len(input.shape))]
    x = xp.pad(input, pad_array, 'constant', constant_values=(0, 0))

    # Fourier space.
    X = xp.fft.fftn(x, axes=(axis,))
    lo = xp.zeros((x.shape[axis],)).astype(xp.complex64)
    lo[:filt_lo.size] = filt_lo
    lo = xp.reshape(xp.fft.fftn(lo, axes=(0,)), [lo.size if k == axis else 1 for k in range(len(x.shape))])
    hi = xp.zeros((x.shape[axis],)).astype(xp.complex64)
    hi[:filt_hi.size] = filt_hi
    hi = xp.reshape(xp.fft.fftn(hi, axes=(0,)), [hi.size if k == axis else 1 for k in range(len(x.shape))])

    # Apply convolutions.
    y_lo = xp.fft.ifftn(X * lo, axes=(axis,))
    y_hi = xp.fft.ifftn(X * hi, axes=(axis,))

    # Sub-sampling
    y_lo = xp.take(y_lo, [t * 2 for t in range(0, y_lo.shape[axis]//2)], axis=axis)
    y_hi = xp.take(y_hi, [t * 2 for t in range(0, y_hi.shape[axis]//2)], axis=axis)

    # Apply recusstion to other axis and concatenate.
    return xp.concatenate((apply_filter_along_axis(y_lo, axes[1:], filt_lo, filt_hi), apply_filter_along_axis(y_hi, axes[1:], filt_lo, filt_hi)), axis=axis)


def fwt(input, wave_name='db4', axes=None, level=None):
    """Forward wavelet transform.

    Args:
        input (array): Input array.
        wave_name (str): Wavelet name.
        axes (None or tuple of int): Axes to perform wavelet transform.
        level (None or int): Number of wavelet levels.
    """
    device = backend.get_device(input)
    xp = device.xp

    if (level == 0):
        return input

    if not axes:
        axes = [k for k in range(len(input.shape)) if input.shape[k] > 1]
        axes = tuple(axes)
    if not level:
        level = pywt.dwt_max_level(xp.min([input.shape[ax] for ax in axes]), dec_hi.size)
    assert(level > 0)

    wavdct = pywt.Wavelet(wave_name)
    dec_hi = xp.array(wavdct.dec_hi)
    dec_lo = xp.array(wavdct.dec_lo)

    y = apply_filter_along_axis(input, axes, dec_lo, dec_hi)

    approx_idx = tuple([slice(0, y.shape[k]//2) if k in axes else slice(0, None) for k in range(len(input.shape))])
    lowlvl = fwt(y[approx_idx], wave_name = wave_name, axes = axes, level = level - 1)

    print("Level %d =================" % level)
    print(input.shape)
    print(y.shape)
    print(y[approx_idx].shape)
    print(lowlvl.shape)
    print("=========================")

    exit(0)
    y[approx_idx] = fwt(y[approx_idx], wave_name = wave_name, axes = axes, level = level - 1)
    
    return y


def fwt_dct(input, wave_name='db4', axes=None, level=None):
    """Forward wavelet transform.

    Args:
        input (array): Input array.
        wave_name (str): Wavelet name.
        axes (None or tuple of int): Axes to perform wavelet transform.
        level (None or int): Number of wavelet levels.
    """
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

    keys = ['%04d' % level]
    for ax in range(len(x.shape)):
        if ax not in axes:
            keys = [elm + 'X' for elm in keys]
        else:
            keys = [elm + 'A' for elm in keys] + [elm + 'D' for elm in keys]

    wav = {}
    f = xp.ones(x.shape).astype(xp.complex)
    for key in keys:
        subkeys = list(key)[4:]
        for k in range(len(subkeys)):
            subkey = subkeys[k]
            if subkey == 'A':
                lo = xp.zeros((x.shape[k],)).astype(xp.complex64)
                lo[:dec_lo.size] = dec_lo
                lo = xp.reshape(xp.fft.fftn(lo, axes=(0,)), \
                        [lo.size if k == s else 1 for s in range(len(x.shape))])
                f = f * lo
            elif subkey == 'D':
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
        approx_key = [elm for elm in wav.keys() if 'D' not in elm].pop()
        approx_val = wav[approx_key]
        next_level = fwt_dct(wav[approx_key], wave_name=wave_name, axes=axes, level=level-1)
        wav.update(next_level)
        wav.pop(approx_key)

    return wav


def iwt_dct(input, oshape, wave_name='db4'):
    """Inverse wavelet transform.

    Args:
        input (array): Input dictionary.
        oshape (tuple of ints): Output shape.
        wave_name (str): Wavelet name.
        axes (None or tuple of int): Axes to perform wavelet transform.
        level (None or int): Number of wavelet levels.
    """
    device = backend.get_device(input[list(input.keys())[0]])
    xp = device.xp

    wavdct = pywt.Wavelet(wave_name)
    rec_hi = xp.array(wavdct.rec_hi)
    rec_lo = xp.array(wavdct.rec_lo)

    max_level = 0;
    for key in input.keys():
        max_level = max(int(key[:4]), max_level)

    axes = get_axes_from_dct(input)

    sampleidx = []
    for k in range(len(oshape)):
        if k in axes:
            sampleidx = sampleidx + [slice(0, None, 2)];
        else:
            sampleidx = sampleidx + [slice(0, None)];
    sampleidx = tuple(sampleidx);

    cropidx = []
    for k in range(len(oshape)):
        if k in axes:
            cropidx = cropidx + [slice(rec_hi.size - 1, oshape[k] + rec_hi.size - 1)];
        else:
            cropidx = cropidx + [slice(0, None)];
    cropidx = tuple(cropidx);

    pad_shape = [(oshape[k] + rec_hi.size - 1 + (oshape[k] + rec_hi.size - 1) % 2) if k in axes else oshape[k] \
                    for k in range(len(oshape))]
    
    inputdct = {}
    approxdct = {}
    for key in list(input.keys()):
        if (int(key[:4]) < max_level):
            approxdct[key] = input[key]
        else:
            inputdct[key] = input[key]

    if approxdct != {}:
        approxkey = [elm for elm in approxdct.keys() if 'D' not in elm].pop()
        approxkey = "%04d%s" % (max_level, approxkey[4:])
        ashape = [pad_shape[k]//2 if k in axes else oshape[k] for k in range(len(oshape))]
        inputdct[approxkey] = iwt_dct(approxdct, ashape, wave_name=wave_name)

    res = xp.zeros(oshape)
    X   = xp.zeros(pad_shape).astype(xp.complex64)
    f   = xp.ones(pad_shape).astype(xp.complex64)
    for key in inputdct.keys():
        X[sampleidx] = inputdct[key];
        X = xp.fft.fftn(X, axes=axes)
        subkeys = list(key)[4:]
        for k in range(len(subkeys)):
            subkey = subkeys[k]
            if subkey == 'A':
                lo = xp.zeros((X.shape[k],)).astype(xp.complex64)
                lo[:rec_lo.size] = rec_lo
                lo = xp.reshape(xp.fft.fftn(lo, axes=(0,)), \
                        [lo.size if k == s else 1 for s in range(len(oshape))])
                f = f * lo
            elif subkey == 'D':
                hi = xp.zeros((X.shape[k],)).astype(xp.complex64)
                hi[:rec_hi.size] = rec_hi
                hi = xp.reshape(xp.fft.fftn(hi, axes=(0,)), \
                        [hi.size if k == s else 1 for s in range(len(oshape))])
                f = f * hi
        y = xp.fft.ifftn(X * f, axes=axes)
        X.fill(0)
        f.fill(1)
        res = res + y[cropidx]

    return res


def iwt(input, oshape, wave_name='db4'):
    """Inverse wavelet transform.

    Args:
        input (array): Input dictionary.
        oshape (tuple of ints): Output shape.
        wave_name (str): Wavelet name.
        axes (None or tuple of int): Axes to perform wavelet transform.
        level (None or int): Number of wavelet levels.
    """
    return iwt_dct(input, oshape, wave_name)
