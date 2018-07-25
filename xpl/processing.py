"""Provides functions for data processing."""
# pylint: disable=invalid-name

import logging

import numpy as np


logger = logging.getLogger(__name__)

def calculate_background(bgtype, energy, intensity):
    """Returns background subtracted intensity."""
    # pylint: disable=unsubscriptable-object
    if bgtype == "linear":
        background = np.linspace(intensity[0], intensity[-1], len(energy))
    elif bgtype == "shirley":
        background = shirley(energy, intensity)
    else:
        background = np.array([0] * len(energy))
    return background


def shirley(energy, intensity, tol=1e-5, maxit=20):
    """Calculates shirley background."""
    if energy[0] < energy[-1]:
        is_reversed = True
        energy = energy[::-1]
        intensity = intensity[::-1]
    else:
        is_reversed = False

    background = np.ones(energy.shape) * intensity[-1]
    integral = np.zeros(energy.shape)
    spacing = (energy[-1] - energy[0]) / (len(energy) - 1)

    subtracted = intensity - background
    ysum = subtracted.sum() - np.cumsum(subtracted)
    for i in range(len(energy)):
        integral[i] = spacing * (ysum[i] - 0.5
                                 * (subtracted[i] + subtracted[-1]))

    iteration = 0
    while iteration < maxit:
        subtracted = intensity - background
        integral = spacing * (subtracted.sum() - np.cumsum(subtracted))
        bnew = ((intensity[0] - intensity[-1])
                * integral / integral[0] + intensity[-1])
        if np.linalg.norm((bnew - background) / intensity[0]) < tol:
            background = bnew.copy()
            break
        else:
            background = bnew.copy()
        iteration += 1
    if iteration >= maxit:
        logger.warning("shirley: Max iterations exceeded before convergence.")

    if is_reversed:
        return background[::-1]
    return background

def smoothen(intensity, interval=20):
    """Smoothed intensity."""
    odd = int(interval / 2) * 2 + 1
    even = int(interval / 2) * 2
    cumsum = np.cumsum(np.insert(intensity, 0, 0))
    avged = (cumsum[odd:] - cumsum[:-odd]) / odd
    for _ in range(int(even / 2)):
        avged = np.insert(avged, 0, avged[0])
        avged = np.insert(avged, -1, avged[-1])
    return avged

def calibrate(energy, shift):
    """Calibrate energy scale."""
    return energy + shift

def x_at_maximum(energy, intensity, span):
    """Calibrate energy axis."""
    emin, emax = span
    idx1, idx2 = sorted([np.searchsorted(energy, emin),
                         np.searchsorted(energy, emax)])
    maxidx = np.argmax(intensity[idx1:idx2]) + idx1
    maxen = energy[maxidx]
    return maxen

def normalize(intensity, norm):
    """Normalize intensity."""
    if not norm:
        return intensity
    if isinstance(norm, (int, float)) and norm != 1:
        normto = norm
    else:
        normto = max(intensity)
    return intensity / normto

def getspan(energy, intensity, eminmax):
    """Get a slice of (energy_s, intensity_s) from energy interval."""
    idx1, idx2 = sorted([
        np.searchsorted(energy, eminmax[0]),
        np.searchsorted(energy, eminmax[1])])
    return energy[idx1:idx2], intensity[idx1:idx2]
