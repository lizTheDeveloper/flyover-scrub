"""Remove aircraft Doppler signatures from audio while keeping the foreground.

Why not just a filter: the attack is a matched filter. It integrates energy
along a long, precisely-predicted curve, so it recovers lines far below the
noise floor and anything merely *attenuated* (a high-pass that leaves the line
20 dB down still scores). A defence has to make the bins carry no information
about the curve, not just make them quieter.

Strategy, per STFT bin inside the protected band:
  * Background bins (near the local noise floor) are *re-synthesised*: replaced
    with fresh random-phase noise drawn at the local floor level. The room tone
    sounds the same, but any sub-floor line is gone, because the floor estimate
    is a median across neighbouring frequencies and a narrow line cannot move it.
  * Foreground bins that form long, slowly-gliding narrowband ridges (aircraft,
    also drones, distant traffic, HVAC) are replaced the same way.
  * Other foreground (speech, music, transients) is passed through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.signal import istft, stft


@dataclass
class ScrubConfig:
    band: tuple[float, float] = (20.0, 2000.0)
    window_s: float = 0.25  # ~4 Hz bins; long enough to separate slow tones from speech
    hop_frac: float = 0.25
    floor_freq_hz: float = 120.0  # width of median across frequency for floor estimate
    floor_time_s: float = 3.0  # smoothing of the floor over time
    foreground_db: float = 12.0  # bins this far above the floor are kept; 8 leaks medium-loud lines
    ridge_db: float = 4.0  # low on purpose: false ridges in pure noise cost nothing (it is resynthesised anyway)
    ridge_min_s: float = 1.5  # ridges at least this long are treated as signatures
    ridge_max_slope_hz_s: float = 40.0  # aircraft glide slowly; speech pitch moves fast
    ridge_pad_hz: float = 8.0
    # Steady tones (mains hum, fridge, machinery) carry no Doppler information, but
    # mains hum enables ENF forensics (timestamping via grid-frequency drift).
    # Removing them is the safer default; set True to keep the room's hum.
    keep_stationary_tones: bool = False
    stationary_max_drift_hz: float = 1.5
    seed: int | None = None


@dataclass
class ScrubReport:
    replaced_fraction: float
    ridge_fraction: float
    ridges_found: int
    extra: dict = field(default_factory=dict)


def _floor(mag: np.ndarray, df: float, dt: float, cfg: ScrubConfig) -> np.ndarray:
    nf = max(3, int(round(cfg.floor_freq_hz / df)) | 1)
    nt = max(1, int(round(cfg.floor_time_s / dt)) | 1)
    logm = np.log(mag + 1e-12)
    fl = ndimage.median_filter(logm, size=(nf, 1), mode="nearest")
    fl = ndimage.uniform_filter1d(fl, size=nt, axis=1, mode="nearest")
    return np.exp(fl)


def _ridges(ratio_db: np.ndarray, df: float, dt: float, cfg: ScrubConfig):
    """Long narrowband ridges with slow frequency drift."""
    peak = (ratio_db > cfg.ridge_db) & (ratio_db >= ndimage.maximum_filter1d(ratio_db, 3, axis=0))
    # allow a couple of bins drift per frame and short dropouts
    max_bins = max(1, int(np.ceil(cfg.ridge_max_slope_hz_s * dt / df)))
    struct = np.ones((2 * max_bins + 1, 3), bool)
    bridged = peak | ndimage.binary_closing(peak, np.ones((1, 3), bool))  # bridge 1-frame dropouts
    lab, n = ndimage.label(bridged, structure=struct)
    if n == 0:
        return np.zeros_like(peak), 0
    objs = ndimage.find_objects(lab)
    min_frames = int(np.ceil(cfg.ridge_min_s / dt))
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        t_extent = sl[1].stop - sl[1].start
        if t_extent < min_frames:
            continue
        # reject components that are really broad (e.g. sustained wideband noise)
        f_extent = sl[0].stop - sl[0].start
        if f_extent * df > cfg.ridge_max_slope_hz_s * t_extent * dt + 6 * df:
            continue
        if cfg.keep_stationary_tones and f_extent * df <= cfg.stationary_max_drift_hz + 2 * df:
            continue
        keep[i] = True
    mask = keep[lab]
    pad = max(1, int(round(cfg.ridge_pad_hz / df)))
    mask = ndimage.binary_dilation(mask, np.ones((2 * pad + 1, 3), bool))
    return mask, int(keep.sum())


def scrub(audio: np.ndarray, sr: int, cfg: ScrubConfig | None = None) -> tuple[np.ndarray, ScrubReport]:
    """Scrub mono ``(n,)`` or multichannel ``(n, ch)`` audio. Returns (audio, report)."""
    cfg = cfg or ScrubConfig()
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        outs, reps = zip(*(scrub(x[:, c], sr, cfg) for c in range(x.shape[1])))
        rep = reps[0]
        rep.extra["per_channel"] = [r.replaced_fraction for r in reps]
        return np.stack(outs, axis=1), rep

    rng = np.random.default_rng(cfg.seed)
    nper = int(round(cfg.window_s * sr))
    nper += nper % 2
    hop = max(1, int(round(nper * cfg.hop_frac)))
    f, t, Z = stft(x, fs=sr, nperseg=nper, noverlap=nper - hop, window="hann")
    df, dt = f[1] - f[0], hop / sr
    mag = np.abs(Z)
    floor = _floor(mag, df, dt, cfg)
    ratio_db = 20 * np.log10((mag + 1e-12) / (floor + 1e-12))

    inband = ((f >= cfg.band[0]) & (f <= cfg.band[1]))[:, None]
    ridge_mask, n_ridges = _ridges(np.where(inband, ratio_db, -99), df, dt, cfg)
    background = ratio_db < cfg.foreground_db
    replace = inband & (background | ridge_mask)

    # Replacement noise is circular complex Gaussian with E|n|^2 = sigma^2.
    #  * background bins: sigma^2 = local *mean power* of the background bins it
    #    replaces (smoothed over floor_freq_hz x floor_time_s). Keeps tonal balance;
    #    a narrow sub-floor line is diluted ~1/(bins in window) and its shape is lost.
    #  * ridge bins: noise at the floor level (Rayleigh median = floor), i.e. removal.
    nf = max(3, int(round(cfg.floor_freq_hz / df)) | 1)
    nt = max(1, int(round(cfg.floor_time_s / dt)) | 1)
    bgm = (inband & background & ~ridge_mask).astype(float)
    num = ndimage.uniform_filter(mag**2 * bgm, size=(nf, nt), mode="nearest")
    den = ndimage.uniform_filter(bgm, size=(nf, nt), mode="nearest")
    p_bg = np.where(den > 1e-3, num / np.maximum(den, 1e-3), floor**2 / np.log(2))
    sigma = np.where(ridge_mask, floor / np.sqrt(np.log(2)), np.sqrt(p_bg))
    noise = sigma * (rng.standard_normal(Z.shape) + 1j * rng.standard_normal(Z.shape)) / np.sqrt(2)
    # Per-frequency level correction for background bins, from whole-file totals
    # only (one scalar per bin, smoothed across frequency): fixes fill bias without
    # reintroducing any time-varying structure.
    bgb = bgm.astype(bool)
    p_in = (mag**2 * bgb).sum(axis=1)
    p_out = (np.abs(noise) ** 2 * bgb).sum(axis=1)
    gain = np.sqrt(np.maximum(ndimage.uniform_filter1d(p_in, nf), 0.0)
                   / np.maximum(ndimage.uniform_filter1d(p_out, nf), 1e-20))
    noise = np.where(bgb, noise * gain[:, None], noise)
    # Never add energy: per frequency (whole-file scalar), replaced bins may not
    # total more power than the originals they replace. Stops edge-of-band bloat.
    p_orig = (mag**2 * replace).sum(axis=1)
    p_new = (np.abs(noise) ** 2 * replace).sum(axis=1)
    cap = np.minimum(1.0, np.sqrt(p_orig / np.maximum(p_new, 1e-30)))
    noise = noise * cap[:, None]
    Zs = np.where(replace, noise, Z)

    _, y = istft(Zs, fs=sr, nperseg=nper, noverlap=nper - hop, window="hann")
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y, ScrubReport(
        replaced_fraction=float(replace[inband[:, 0]].mean()),
        ridge_fraction=float((ridge_mask & inband)[inband[:, 0]].mean()),
        ridges_found=n_ridges,
    )


@dataclass
class RidgeInfo:
    t_start: float
    t_end: float
    f_start: float
    f_end: float
    strength_db: float


def detect_ridges(audio: np.ndarray, sr: int, cfg: ScrubConfig | None = None, *, min_db: float = 8.0,
                  min_s: float = 20.0, min_glide: float = 0.01, window_s: float = 1.0) -> list[RidgeInfo]:
    """Long, *gliding* narrowband ridges (aircraft-like), strongest first.

    Unlike the scrub's internal ridge mask (tuned to over-detect, since a false ridge
    costs nothing there), this is a reporting detector: a ridge must last ``min_s``
    seconds, stand ``min_db`` above the local floor, and change frequency by at least
    ``min_glide`` (fractional) end to end, so steady hum and machinery don't count.
    Lines below the floor are not found here, though a matched filter can still
    recover them: absence of ridges is not proof of absence.
    """
    cfg = cfg or ScrubConfig()
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    nper = int(round(window_s * sr))
    nper += nper % 2
    hop = max(1, nper // 4)
    f, t, Z = stft(x, fs=sr, nperseg=nper, noverlap=nper - hop, window="hann")
    df, dt = f[1] - f[0], hop / sr
    mag = np.abs(Z)
    ratio_db = 20 * np.log10((mag + 1e-12) / (_floor(mag, df, dt, cfg) + 1e-12))
    inband = ((f >= cfg.band[0]) & (f <= cfg.band[1]))[:, None]
    r = np.where(inband, ratio_db, -99)
    peak = (r > min_db) & (r >= ndimage.maximum_filter1d(r, 3, axis=0))
    max_bins = max(1, int(np.ceil(5.0 * dt / df)))  # aircraft glide < ~5 Hz/s
    bridged = peak | ndimage.binary_closing(peak, np.ones((1, 5), bool))
    bridged = ndimage.binary_dilation(bridged, np.ones((2 * max_bins + 1, 1), bool))  # allow glide between frames
    lab, n = ndimage.label(bridged, structure=np.ones((3, 3), bool))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or (sl[1].stop - sl[1].start) * dt < min_s:
            continue
        comp = (lab == i) & peak
        cols = np.where(comp.any(axis=0))[0]
        if len(cols) * dt < 0.6 * min_s:
            continue
        fr = np.array([np.average(f[comp[:, c]], weights=mag[comp[:, c], c]) for c in cols])
        k = max(3, len(cols) // 10)
        f0, f1 = float(np.median(fr[:k])), float(np.median(fr[-k:]))
        if abs(f1 - f0) < min_glide * (f0 + f1) / 2:
            continue
        out.append(RidgeInfo(float(t[cols[0]]), float(t[cols[-1]]), f0, f1, float(np.median(r[comp]))))
    return sorted(out, key=lambda z: -z.strength_db)
