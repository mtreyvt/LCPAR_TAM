# TimeGating.py
import numpy as np
from scipy.signal import tukey, savgol_filter
import pywt

def impulse_response(freq_resp: np.ndarray, N_fft: int) -> np.ndarray:
    """
    Convert frequency response (angles x freqs) to impulse response (angles x time).
    freq_resp: complex array [N_angles, N_freqs]
    """
    return np.fft.ifft(freq_resp, n=N_fft, axis=1)

def _find_gate_center_idx(h_t: np.ndarray, fs: float) -> int:
    """
    Pick the direct-path arrival by finding the strongest *early* peak,
    using angle-averaged power to avoid per-angle outliers.
    """
    N_angles, N_time = h_t.shape
    early = max(1, N_time // 4)     # search in first quarter by default
    power_mean = np.mean(np.abs(h_t[:, :early])**2, axis=0)
    return int(np.argmax(power_mean))  # time index

def apply_time_gate(h_t: np.ndarray, fs: float, gate_ns: float = 10.0, alpha: float = 0.5,
                    center_idx: int | None = None) -> np.ndarray:
    """
    Apply a Tukey window time-gate around the strongest early peak (or a provided center index).

    h_t: impulse response [angles, time]
    fs:  sample rate (Hz)
    gate_ns: gate width in nanoseconds
    alpha: Tukey shape parameter (0=rect, 1=Hann)
    center_idx: optional index to center the gate (overrides auto-pick)
    """
    N_angles, N_time = h_t.shape

    # gate length in samples (>=1)
    gate_len = max(1, int(np.ceil((gate_ns * 1e-9) * fs)))
    if center_idx is None:
        center_idx = _find_gate_center_idx(h_t, fs)

    start = max(0, center_idx - gate_len // 2)
    end   = min(N_time, start + gate_len)
    if end <= start:
        start, end = 0, min(N_time, gate_len)

    win = tukey(end - start, alpha=alpha) if (end - start) > 1 else np.ones(1)
    g = np.zeros(N_time, dtype=float)
    g[start:end] = win

    return h_t * g[np.newaxis, :]

def gated_frequency_response(h_t_gated: np.ndarray, N_fft: int) -> np.ndarray:
    """FFT back to frequency domain after time gating."""
    return np.fft.fft(h_t_gated, n=N_fft, axis=1)

def extract_pattern(H_gated: np.ndarray) -> np.ndarray:
    """
    Extract polar pattern in dB from the gated frequency response.
    Uses magnitude of the DC bin (bin 0).
    """
    mags = np.abs(H_gated[:, 0])          # DC bin
    mags = mags / (np.max(mags) if np.max(mags) > 0 else 1.0)
    return 20.0 * np.log10(np.clip(mags, 1e-12, None))

def denoise_pattern(pattern_db: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """
    Smooth jagged gated pattern using Savitzky-Golay filter (lightweight, no pywt needed).
    """
    if len(pattern_db) < window:
        return pattern_db
    return savgol_filter(pattern_db, window, poly)
    
    
def denoise_pattern_wavelet(pattern_db: np.ndarray,
                            wavelet: str = "db4",
                            level: int = 3,
                            thresh_scale: float = 0.5) -> np.ndarray:
    """
    Denoise antenna pattern using wavelet thresholding.
    Removes multipath-related ripples while preserving main lobes.
    """
    if len(pattern_db) < 4:
        return pattern_db

    max_level = min(level, pywt.dwt_max_level(len(pattern_db), pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(pattern_db, wavelet, level=max_level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = thresh_scale * sigma * np.sqrt(2 * np.log(len(pattern_db)))

    coeffs_thresh = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    pattern_clean = pywt.waverec(coeffs_thresh, wavelet)

    # Match input length
    return pattern_clean[:len(pattern_db)]

def apply_denoise_methods(pattern_db: np.ndarray,
                          use_wavelet: bool = False,
                          use_savgol: bool = True,
                          window: int = 11,
                          poly: int = 3) -> np.ndarray:
    """
    Apply one or both denoising methods in sequence:
      - Wavelet denoise (multi-scale thresholding)
      - Savitzky-Golay smoothing (light polynomial filter)

    Set flags to control the combination.
    """
    pat = np.copy(pattern_db)
    if use_wavelet:
        pat = denoise_pattern_wavelet(pat)
    if use_savgol:
        pat = denoise_pattern(pat, window=window, poly=poly)
    return pat


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def _infer_fs_from_freqs(freq_list: np.ndarray, N_fft: int) -> float:
    """
    Infer a notional sample rate fs from frequency grid spacing:
      df = fs / N_fft  =>  fs = df * N_fft
    """
    f = np.asarray(freq_list, dtype=float).ravel()
    if f.size < 2:
        raise ValueError("Need at least 2 frequency points to infer fs; provide fs explicitly.")
    df = float(np.median(np.diff(np.sort(f))))
    return df * float(N_fft)

def apply_time_gating_matrix(freq_resp: np.ndarray,
                             freq_list: np.ndarray,
                             gate_width_s: float = 25e-9,
                             *,
                             fs: float | None = None,
                             tukey_alpha: float = 0.5,
                             N_fft: int | None = None,
                             denoise_mode: str = "savgol") -> np.ndarray:
    """
    Convenience helper used by RadioFunctions:
      1) zero-pad to N_fft
      2) IFFT to time domain
      3) time-gate with Tukey window (width=gate_width_s)
      4) FFT back
      5) take DC bin magnitude → dB
      6) optional denoise (Savitzky-Golay, wavelet, or both)

    freq_resp: complex matrix [N_angles, N_freqs]
    freq_list: array-like of frequency points (Hz) used to build the columns
    gate_width_s: gate width in seconds (e.g., 25e-9 for 25 ns)
    fs: optional sample rate to use for gating timeline; if None, infer from df
    tukey_alpha: Tukey window alpha
    N_fft: optional FFT size; if None, choose next pow2 ≥ 2×N_freqs
    denoise_mode: "none", "savgol", "wavelet", or "both"

    Returns: pattern_dB, shape [N_angles]
    """
    # Ensure proper type/shape
    freq_resp = np.asarray(freq_resp, dtype=np.complex128)
    if freq_resp.ndim != 2:
        raise ValueError("freq_resp must be a 2D array [N_angles, N_freqs].")

    N_angles, N_freqs = freq_resp.shape
    if N_fft is None:
        N_fft = _next_pow2(max(256, 2 * N_freqs))

    if fs is None:
        fs = _infer_fs_from_freqs(np.asarray(freq_list, float), N_fft)

    # --- Time-gating pipeline ---
    h_t = impulse_response(freq_resp, N_fft)                                  # [N_angles, N_fft]
    h_t_g = apply_time_gate(h_t, fs, gate_ns=(gate_width_s * 1e9), alpha=tukey_alpha)
    H_g = gated_frequency_response(h_t_g, N_fft)                              # [N_angles, N_fft]
    pat_db = extract_pattern(H_g)                                             # [N_angles]

    # --- Denoising section ---
    denoise_mode = denoise_mode.lower().strip()
    if denoise_mode not in {"none", "savgol", "wavelet", "both"}:
        raise ValueError(f"Invalid denoise_mode: '{denoise_mode}'. Must be one of "
                         "'none', 'savgol', 'wavelet', 'both'.")

    if denoise_mode == "savgol":
        pat_db = apply_denoise_methods(pat_db, use_wavelet=False, use_savgol=True)
    elif denoise_mode == "wavelet":
        pat_db = apply_denoise_methods(pat_db, use_wavelet=True, use_savgol=False)
    elif denoise_mode == "both":
        pat_db = apply_denoise_methods(pat_db, use_wavelet=True, use_savgol=True)
    # "none" skips all denoising

    # --- Normalize output ---
    pat_db = pat_db - np.max(pat_db) if pat_db.size else pat_db
    return pat_db

def print_and_return_data(data):
    """
    Legacy helper for debugging.
    Just print length and return unchanged.
    """
    arr = np.asarray(data, dtype=float)
    print(f"[TimeGating] Data length: {arr.size}, dtype={arr.dtype}")
    return arr
