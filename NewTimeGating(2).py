"""NewTimeGating
=================

This module implements a simplified time‑gating algorithm inspired by the
automatic calibration procedure described in Bekasiewicz et al. (2018)
for far‑field measurements in non‑anechoic environments.  The classic
approach uses a Tukey window centred on the direct‑path peak of the
impulse response.  The new method implemented here instead derives a
rectangular gate from the envelope of the impulse response itself.  A
threshold is set as a fraction of the peak power and the gate spans
the contiguous region where the average power remains above this
threshold.  The rectangular gate naturally adapts its width to the
dominant arrival and can better reject late reflections when the
measurement environment is highly reflective【905336941599853†L420-L505】.

The functions in this file mirror those provided in ``TimeGating.py``
but use a rectangular window determined by a threshold rather than a
Tukey taper.  They are designed to be lightweight and have no
external dependencies beyond NumPy.  See ``RadioFunctions.do_time_sync_tg_new``
for an example of how to use them in a measurement pipeline.
"""

from __future__ import annotations

import numpy as np


def impulse_response(freq_resp: np.ndarray, N_fft: int) -> np.ndarray:
    """
    Compute the impulse response of a set of frequency sweeps.

    Given a matrix of complex frequency responses of shape
    ``(N_angles, N_freqs)`` this helper zero‑pads each row to length
    ``N_fft`` and performs an IFFT along the frequency axis.  The
    resulting impulse response has shape ``(N_angles, N_fft)``.  The
    output is complex because the input response generally has both
    magnitude and phase.

    Parameters
    ----------
    freq_resp : np.ndarray
        Complex matrix of shape ``(N_angles, N_freqs)``.
    N_fft : int
        Length of the FFT used for zero‑padding and interpolation.  A
        power‑of‑two value is recommended for efficient FFTs.

    Returns
    -------
    np.ndarray
        Complex matrix containing the impulse response per angle, of
        shape ``(N_angles, N_fft)``.
    """
    return np.fft.ifft(freq_resp, n=N_fft, axis=1)


def _next_pow2(n: int) -> int:
    """Return the next power of two greater than or equal to ``n``."""
    p = 1
    while p < n:
        p <<= 1
        
    return p

def _infer_fs_from_freqs(freq_list: np.ndarray, N_fft: int) -> float:
    """
    Infer a nominal sampling rate for the impulse response from the
    frequency grid.

    The frequency sweep used to measure the antenna response is assumed
    to be uniformly spaced.  The sampling rate in the time domain,
    ``fs``, must satisfy the relationship ``fs = df * N_fft``, where
    ``df`` is the frequency step.  This helper computes the median
    spacing of the sorted frequency list and multiplies it by
    ``N_fft``.  At least two frequency points are required to
    estimate ``df``.

    Parameters
    ----------
    freq_list : np.ndarray
        One‑dimensional array of frequency values in Hz.  Only the
        ordering and spacing matter; duplicates are ignored by the
        median operation.
    N_fft : int
        FFT length used when converting to the time domain.

    Returns
    -------
    float
        Estimated sampling rate in Hz for the impulse response.

    Raises
    ------
    ValueError
        If fewer than two distinct frequency points are provided.
    """
    f = np.asarray(freq_list, dtype=float).ravel()
    if f.size < 2:
        raise ValueError(
            "Need at least 2 frequency points to infer fs automatically; "
            "please supply fs explicitly."
        )
    # Sort frequencies and compute median of adjacent differences
    df = float(np.median(np.diff(np.sort(f))))
    return df * float(N_fft)


def refine_gate_greedy(
    h_t: np.ndarray,
    fs: float,
    *,
    start_idx: int | None = None,
    width_s: float = 25e-9,
    thr_factor: float = 0.5,
    step_s: float = 0.5e-9,
    iters: int = 20,
) -> tuple[int, int]:
    """
    Greedily refine the rectangular gate boundaries to minimise pattern
    variance across angles.

    This routine provides a simple unsupervised heuristic for
    fine‑tuning the start and stop indices of a rectangular time gate.
    The goal is to select a window that yields a stable across‑angle
    magnitude when transforming back to the frequency domain.  It is
    particularly useful when no reference pattern is available for
    calibration but one wishes to suppress multipath while preserving
    the direct path.

    The algorithm proceeds as follows:

    1. Compute an initial gate using the mean power envelope and the
       threshold ``thr_factor`` (as in ``rectangular_gate``).  If
       ``start_idx`` is provided the initial start index is forced
       accordingly.
    2. Define a score as the variance of the DC‑bin magnitudes of
       ``H = FFT(h_t * gate)`` across angles.  Lower variance implies
       more stable patterns.
    3. Perturb the start and stop indices by ±``step_s`` seconds (in
       samples) and accept any change that reduces the score.  The
       search is greedy: the first improvement is taken and the
       process repeats until no improvement is found.  The step size
       is reduced by half when no improvement occurs in a pass.
    4. After ``iters`` iterations or convergence, return the best
       start and stop indices.

    Parameters
    ----------
    h_t : np.ndarray
        Complex impulse response of shape ``(N_angles, N_time)``.
    fs : float
        Sampling rate (Hz) used to compute the time axis.
    start_idx : int, optional
        Optional initial start index for the gate.  If ``None``, the
        start index is determined from the threshold method.
    width_s : float, optional
        Initial gate width in seconds when no thresholded region is
        found.  Defaults to 25 ns.
    thr_factor : float, optional
        Fraction of the peak mean power used to threshold the envelope.
    step_s : float, optional
        Step size in seconds for adjusting the gate boundaries.  A
        smaller value yields finer adjustments at the expense of more
        iterations.  Defaults to 0.5 ns.
    iters : int, optional
        Maximum number of greedy iterations.  Defaults to 20.

    Returns
    -------
    tuple[int, int]
        A pair ``(start_idx, stop_idx)`` giving the best gate
        boundaries in samples.  These indices can be used to build a
        rectangular window: ``gate[start_idx:stop_idx] = 1``.
    """
    N_time = h_t.shape[1]
    # Compute initial gate using threshold on mean envelope
    env = np.mean(np.abs(h_t) ** 2, axis=0)
    max_env = float(np.max(env)) if np.max(env) > 0 else 1e-12
    threshold = float(thr_factor) * max_env
    above = np.where(env >= threshold)[0]
    if above.size:
        s0, e0 = int(above[0]), int(above[-1]) + 1
    else:
        # fallback around strongest peak
        center = int(np.argmax(env))
        width_samples = max(1, int(np.ceil(width_s * fs)))
        s0 = max(0, center - width_samples // 2)
        e0 = min(N_time, s0 + width_samples)
    if start_idx is not None:
        s0 = int(np.clip(int(start_idx), 0, N_time - 1))
        e0 = int(np.clip(s0 + max(1, int(np.ceil(width_s * fs))), s0 + 1, N_time))

    # Define scoring function: variance of DC bin magnitudes
    def score(s: int, e: int) -> float:
        gate = np.zeros(N_time, dtype=float)
        gate[s:e] = 1.0
        H = np.fft.fft(h_t * gate[np.newaxis, :], axis=1)
        dc = np.abs(H[:, 0])
        dc /= (np.max(dc) if np.max(dc) > 0 else 1.0)
        return float(np.var(dc))

    best_s, best_e = s0, e0
    best_score = score(best_s, best_e)
    step_samples = max(1, int(round(step_s * fs)))
    for _ in range(int(iters)):
        improved = False
        # try adjusting start and end by ±step
        for ds, de in ((-step_samples, 0), (step_samples, 0), (0, -step_samples), (0, step_samples)):
            s = int(np.clip(best_s + ds, 0, N_time - 2))
            e = int(np.clip(best_e + de, s + 1, N_time))
            sc = score(s, e)
            if sc < best_score:
                best_s, best_e, best_score = s, e, sc
                improved = True
                break
        if not improved:
            if step_samples > 1:
                step_samples = max(1, step_samples // 2)
            else:
                break
    return best_s, best_e


def rectangular_gate(h_t: np.ndarray, fs: float, *, gate_ns: float = 10.0,
                     thr_factor: float = 0.5) -> np.ndarray:
    """Apply a rectangular time gate to the impulse response.

    The gate is determined by first computing the average power envelope
    across all angles.  A threshold is defined as ``thr_factor`` times
    the maximum average power.  The gate spans the contiguous region
    where the envelope exceeds this threshold.  If no samples exceed
    the threshold (e.g. due to noise), a fixed window of width
    ``gate_ns`` nanoseconds centred on the strongest peak is used.

    Parameters
    ----------
    h_t : np.ndarray
        Impulse response, shape (N_angles, N_time).
    fs : float
        Sampling rate in Hz used to compute the impulse response.
    gate_ns : float, optional
        Default width of the gate in nanoseconds when no thresholded
        region is found.  This value is converted to a sample count.
    thr_factor : float, optional
        Fraction of the peak average power used as the threshold.  A
        value between 0 and 1.  A typical value of 0.5 captures the
        main lobe around the direct‑path arrival【905336941599853†L420-L505】.

    Returns
    -------
    np.ndarray
        Time‑gated impulse response with the same shape as ``h_t``.
    """
    N_angles, N_time = h_t.shape
    # Compute the average power envelope across angles
    env = np.mean(np.abs(h_t)**2, axis=0)
    max_env = float(np.max(env))
    if not np.isfinite(max_env) or max_env <= 0.0:
        max_env = 1e-12
    threshold = thr_factor * max_env
    # Find contiguous region above threshold
    above = np.where(env >= threshold)[0]
    if above.size > 0:
        start = int(above[0])
        end = int(above[-1]) + 1  # include last sample
    else:
        # Fall back to a fixed window around the strongest peak
        center = int(np.argmax(env))
        gate_len = max(1, int(np.ceil((gate_ns * 1e-9) * fs)))
        start = max(0, center - gate_len // 2)
        end = min(N_time, start + gate_len)
    # Create rectangular window
    gate = np.zeros(N_time, dtype=float)
    gate[start:end] = 1.0
    return h_t * gate[np.newaxis, :]


def gated_frequency_response(h_t_gated: np.ndarray, N_fft: int) -> np.ndarray:
    """FFT back to the frequency domain after time gating."""
    return np.fft.fft(h_t_gated, n=N_fft, axis=1)


def extract_pattern(H_gated: np.ndarray) -> np.ndarray:
    """Extract polar pattern in dB from the gated frequency response.

    Uses the magnitude of the DC bin (bin 0) as the per‑angle value and
    normalises the resulting pattern such that its peak is at 0 dB.
    """
    mags = np.abs(H_gated[:, 0])
    mags = mags / (np.max(mags) if np.max(mags) > 0 else 1.0)
    pattern_db = 20.0 * np.log10(np.clip(mags, 1e-12, None))
    if pattern_db.size:
        pattern_db = pattern_db - np.max(pattern_db)
    return pattern_db


def apply_time_gating_matrix_rect(
    freq_resp: np.ndarray,
    freq_list: np.ndarray,
    *,
    gate_width_s: float = 25e-9,
    fs: float | None = None,
    thr_factor: float = 0.5,
    N_fft: int | None = None,
) -> np.ndarray:
    """Convenience wrapper to time‑gate a frequency response using a
    rectangular window.

    The pipeline consists of:

    1. Zero‑pad the frequency response to ``N_fft`` and take the IFFT
       to obtain the impulse response.
    2. Apply a rectangular gate determined either by the threshold
       ``thr_factor`` or by a default width ``gate_width_s``.
    3. FFT back to the frequency domain and extract the DC bin
       magnitude in decibels.
    4. Normalise so that the peak of the pattern is 0 dB.

    Parameters
    ----------
    freq_resp : np.ndarray
        Complex matrix of shape (N_angles, N_freqs) containing the
        measured frequency response for each angle.
    freq_list : np.ndarray
        1‑D array of frequency points (Hz) used to construct the
        columns of ``freq_resp``.
    gate_width_s : float, optional
        Default gate width in seconds to use when no thresholded
        region is detected.
    fs : float, optional
        Sampling rate (Hz).  If ``None``, the rate is inferred from the
        frequency spacing.
    thr_factor : float, optional
        Fraction of the peak average power used as the threshold.
    N_fft : int, optional
        FFT size.  If ``None``, the next power of two of twice the
        number of frequency points is chosen.

    Returns
    -------
    np.ndarray
        1‑D array of length ``N_angles`` containing the gated pattern in
        dB, normalised so the peak is at 0 dB.
    """
    # Normalise input shape and type
    freq_resp = np.asarray(freq_resp, dtype=np.complex128)
    if freq_resp.ndim != 2:
        raise ValueError("freq_resp must be a 2D array [N_angles, N_freqs].")
    N_angles, N_freqs = freq_resp.shape

    # If only a single frequency is provided we cannot perform time
    # domain gating.  In this degenerate case return a normalised
    # magnitude pattern so the caller still receives a sane result.
    if N_freqs < 2:
        mags = np.abs(freq_resp[:, 0])
        denom = np.max(mags) if np.max(mags) > 0 else 1.0
        mags = mags / denom
        pat_db = 20.0 * np.log10(np.clip(mags, 1e-12, None))
        if pat_db.size:
            pat_db = pat_db - np.max(pat_db)
        return pat_db

    # Choose FFT length if not provided.  A modest minimum of 16 points
    # ensures the impulse response has at least one full cycle even for
    # small frequency sweeps.  Otherwise use twice the number of
    # frequencies and round up to a power of two to avoid circular
    # convolution when gating.
    if N_fft is None:
        N_fft = _next_pow2(max(16, 2 * N_freqs))

    # Infer sampling rate from the frequency spacing if not supplied.  If
    # fewer than two frequencies are present this will error, but the
    # single‑frequency case is handled above.
    if fs is None:
        fs = _infer_fs_from_freqs(np.asarray(freq_list, dtype=float), N_fft)

    # Convert the complex response to an impulse response
    h_t = impulse_response(freq_resp, N_fft)
    # Apply a rectangular time gate.  gate_width_s is in seconds; convert
    # to nanoseconds for compatibility with the rectangular_gate API.
    h_t_g = rectangular_gate(
        h_t,
        fs,
        gate_ns=float(gate_width_s) * 1e9,
        thr_factor=float(thr_factor),
    )
    # Transform back to the frequency domain
    H_g = gated_frequency_response(h_t_g, N_fft)
    # Extract the per‑angle pattern in dB and normalise to peak 0 dB
    pat_db = extract_pattern(H_g)
    return pat_db