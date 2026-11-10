import numpy as np

from flyover_scrub.attack import score_candidates, whitened_spectrogram
from flyover_scrub.physics import (
    SPEED_OF_SOUND,
    emission_geometry,
    make_track,
    received_doppler,
    synthesize_flyover,
)
from flyover_scrub.scrub import ScrubConfig, scrub

SR = 8000


def _straight(v=80.0):
    return make_track(0.0, 200.0, start_xy=(-8000.0, 500.0), altitude=600.0, speed=v, heading_deg=0.0)


def test_doppler_limits_match_textbook():
    trk = _straight(80.0)
    _, g, _ = emission_geometry(trk, np.zeros(3))
    # far approach / far recede -> c/(c -/+ v) for a source moving straight at you
    assert abs(g[5] - SPEED_OF_SOUND / (SPEED_OF_SOUND - 80.0)) < 0.01
    assert abs(g[-5] - SPEED_OF_SOUND / (SPEED_OF_SOUND + 80.0)) < 0.01


def test_emission_time_is_not_just_a_delay():
    """The naive model (velocity at reception time) gives a differently shaped curve."""
    trk = _straight()
    obs = np.zeros(3)
    t_r = np.linspace(80, 130, 200)
    g_right, _, _ = received_doppler(trk, obs, t_r)
    _, g_e, _ = emission_geometry(trk, obs)
    g_naive = np.interp(t_r, trk.t, g_e)
    best = min(np.abs(np.interp(t_r + s, t_r, g_naive) - g_right).max() for s in np.linspace(-10, 10, 201))
    assert best > 0.01  # no time shift reconciles them


def test_synth_puts_energy_on_predicted_curve():
    trk = _straight()
    obs = np.zeros(3)
    n = SR * 60
    x = synthesize_flyover(trk, obs, 70.0, n, SR, f0=100.0, broadband=0.0)
    spec = whitened_spectrogram(x + 1e-3 * np.random.default_rng(0).normal(size=n), SR)
    g, _, _ = received_doppler(trk, obs, 70.0 + spec.times)
    bins = np.rint(100.0 * g / spec.df).astype(int)
    on = spec.S[bins, np.arange(len(bins))].mean()
    off = spec.S[bins + 15, np.arange(len(bins))].mean()
    assert on > off + 2.0


def test_attack_prefers_truth_and_scrub_removes_it():
    rng = np.random.default_rng(3)
    trk = make_track(0.0, 260.0, start_xy=(-9000.0, -400.0), altitude=700.0, speed=70.0, heading_deg=5.0,
                     turn_rate_deg_s=-1.0, turn_start=100.0, turn_duration=25.0)
    obs = np.zeros(3)
    n = SR * 80
    plane = synthesize_flyover(trk, obs, 100.0, n, SR, f0=90.0, rng=rng)
    bg = rng.normal(size=n)
    plane *= 10 ** (-28 / 20) * bg.std() / plane[plane != 0].std()
    mix = bg + plane
    cands = np.array([[0, 0, 0], [0, 800, 0], [600, -600, 0], [-1500, 900, 0]], float)
    kw = dict(f0_range=(87.0, 93.0), offsets=np.arange(-2.0, 2.01, 0.25))

    res = score_candidates(whitened_spectrogram(mix, SR), [trk], 100.0, cands, **kw)
    assert np.argmax(res.scores) == 0

    y, _ = scrub(mix, SR, ScrubConfig(seed=0))
    res2 = score_candidates(whitened_spectrogram(y, SR), [trk], 100.0, cands, **kw)
    ctrl = score_candidates(whitened_spectrogram(bg, SR), [trk], 100.0, cands, **kw)
    # After scrubbing, the true location must score like it does with no aircraft
    # at all. (Comparing argmax over 4 candidates is meaningless: geometry alone
    # can make truth #1 even in the control.)
    assert res.scores[0] - ctrl.scores[0] > 0.3
    assert res2.scores[0] - ctrl.scores[0] < 0.03


def test_scrub_preserves_foreground_and_level():
    rng = np.random.default_rng(1)
    n = SR * 20
    t = np.arange(n) / SR
    bg = 0.01 * rng.normal(size=n)
    # bursty "speech-like" harmonic content: short syllables, fast pitch movement
    env = (np.sin(2 * np.pi * 2.5 * t) > 0.3).astype(float)
    f0 = 180 + 40 * np.sin(2 * np.pi * 3.0 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    voice = env * sum(0.2 / k * np.sin(k * ph) for k in range(1, 8))
    y, rep = scrub(bg + voice, SR, ScrubConfig(seed=0))
    assert rep.ridges_found <= 3  # syllables must not be chained into long ridges
    on = env > 0
    err = np.std(y[on] - (bg + voice)[on]) / np.std(voice[on])
    assert err < 0.35
    assert abs(20 * np.log10(np.std(y) / np.std(bg + voice))) < 1.0


def test_adsb_log_roundtrip(tmp_path):
    """A log in adsb.lol format comes back as a track close to the true path."""
    import json, math
    from flyover_scrub.adsb import FT, load_tracks

    lat0, lon0, R = 41.9, -87.7, 6371008.8
    trk = make_track(1000.0, 200.0, start_xy=(-6000.0, 300.0), altitude=800.0, speed=70.0, heading_deg=0.0)
    rng = np.random.default_rng(0)
    lines = [json.dumps({"meta": {}})]
    for now in np.arange(1003.0, 1197.0, 5.0):
        seen = float(rng.uniform(0, 2))
        p = trk.resample(np.array([now - seen])).pos[0]
        lines.append(json.dumps({"now": now, "ac": [{
            "hex": "a1", "flight": "T1", "t": "C172", "category": "A1",
            "lat": lat0 + math.degrees(p[1] / R),
            "lon": lon0 + math.degrees(p[0] / (R * math.cos(math.radians(lat0)))),
            "alt_geom": p[2] / FT, "alt_baro": None, "seen_pos": seen}]}))
        lines.append(json.dumps({"now": now, "ac": [{"hex": "gnd", "lat": lat0, "lon": lon0, "alt_baro": "ground"}]}))
    log = tmp_path / "f.jsonl"
    log.write_text("\n".join(lines))
    tracks = load_tracks(log, lat0, lon0)
    assert set(tracks) == {"a1"}  # aircraft on the ground are skipped
    got = tracks["a1"]["track"]
    ref = trk.resample(got.t[(got.t > 1020) & (got.t < 1180)])
    err = np.linalg.norm(got.resample(ref.t).pos - ref.pos, axis=1).max()
    assert err < 15.0


def test_trace_fit_finds_truth_and_rejects_reversal():
    from flyover_scrub.trace import locate_trace, reversal_ratio, trace_ridge

    rng = np.random.default_rng(5)
    trk = make_track(0.0, 260.0, start_xy=(-9000.0, -400.0), altitude=600.0, speed=60.0, heading_deg=5.0)
    obs = np.zeros(3)
    sr = 16000
    n = sr * 120
    x = synthesize_flyover(trk, obs, 90.0, n, sr, f0=80.0, rng=rng) * 0.15 + rng.normal(size=n)
    spec = whitened_spectrogram(x, sr, window_s=2.0)
    ridge = trace_ridge(spec, f_range=(65.0, 100.0))
    assert len(ridge) >= 40
    res = locate_trace(ridge, trk, 90.0, (700.0, -500.0), half_width=2500.0, coarse=250.0, fine=50.0)
    s = res.summary(np.zeros(2))
    assert s["top_error_m"] < 400
    assert reversal_ratio(ridge, trk, 90.0, res) > 3


def test_enf_detected_in_grid_hum_and_removed_by_scrub():
    from flyover_scrub.enf import detect_enf

    sr = 8000
    n = sr * 90
    t = np.arange(n) / sr
    rng = np.random.default_rng(2)
    # grid frequency wandering by a few hundredths of a hertz around 60 Hz
    wander = 60 + 0.02 * np.sin(2 * np.pi * t / 37) + 0.01 * np.sin(2 * np.pi * t / 11)
    phase = 2 * np.pi * np.cumsum(wander) / sr
    hum = 0.05 * np.sin(2 * phase) + 0.03 * np.sin(4 * phase)  # 120 and 240 Hz, as transformers hum
    noise = rng.normal(size=n) * 0.3
    res = detect_enf(hum + noise, sr)
    assert res.present and res.nominal_hz == 60
    assert abs(res.grid_hz - 60) < 0.01
    assert not detect_enf(noise, sr).present
    # a machine tone that drifts independently at each "harmonic" is not ENF
    fan = 0.05 * np.sin(2 * np.pi * 120.3 * t) + 0.03 * np.sin(2 * np.pi * 239.1 * t)
    assert not detect_enf(fan + noise, sr).present
    y, _ = scrub(hum + noise, sr, ScrubConfig(seed=0))
    assert not detect_enf(y, sr).present


def test_detect_ridges_finds_flyover_not_noise_or_hum():
    from flyover_scrub.scrub import detect_ridges

    rng = np.random.default_rng(4)
    trk = make_track(0.0, 260.0, start_xy=(-9000.0, -400.0), altitude=700.0, speed=70.0, heading_deg=5.0)
    n = SR * 120
    bg = rng.normal(size=n)
    plane = synthesize_flyover(trk, np.zeros(3), 70.0, n, SR, f0=90.0, rng=rng)
    plane *= 10 ** (-10 / 20) * bg.std() / plane[plane != 0].std()
    ridges = detect_ridges(bg + plane, SR)
    assert len(ridges) >= 1
    assert ridges[0].f_start > ridges[0].f_end  # a flyover falls in pitch
    assert detect_ridges(bg, SR) == []
    t = np.arange(n) / SR
    hum = 0.2 * np.sin(2 * np.pi * 60 * t) + 0.1 * np.sin(2 * np.pi * 120 * t)
    assert detect_ridges(bg + hum, SR) == []
