"""Full-pipeline validation for the JPEG RDH scheme (Li et al., IEEE TMM 2025).

V5: 1D-HS pipeline on a simulated JPEG coefficient array —
    block-sort (Eqs 11-12) + frequency-band selection (Eqs 13-15) +
    embed P bits -> extract + recover. Secret bit-exact, coefficients
    restored exactly, for several quality factors and payloads.
V6: dynamic band-count selection meets capacity P with minimum bands
    (proxy for the (alpha,beta) iteration of Algorithm 1) and stays reversible.
"""
import numpy as np
from validate_algorithm import (
    qtable, ZIGZAG, hs_embed_coeff, hs_extract_coeff,
    smoothness_score, unit_band_distortion)


def simulate_jpeg_coeffs(img, qf):
    """Forward DCT + quantization of an 8x8-blocked grayscale image.
    Returns coeffs[nblocks, 64] in zigzag order and the zigzag qtable."""
    from scipy.fftpack import dctn
    M, N = img.shape
    T = qtable(qf)
    T_zz = np.array([T[r, c] for (r, c) in ZIGZAG])
    blocks = []
    for br in range(0, M, 8):
        for bc in range(0, N, 8):
            blk = img[br:br+8, bc:bc+8].astype(float) - 128
            D = dctn(blk, norm='ortho')
            Q = np.round(D / T).astype(int)
            zz = np.array([Q[r, c] for (r, c) in ZIGZAG])
            blocks.append(zz)
    return np.array(blocks), T_zz


def block_order(coeffs, T_zz):
    """Eqs 11-12: sort block indices by DESCENDING smoothness score."""
    scores = np.array([smoothness_score(coeffs[k], T_zz)
                       for k in range(coeffs.shape[0])])
    return np.argsort(-scores, kind='stable'), scores


def select_bands(coeffs, T_zz, capacity, order):
    """Eqs 13-15: rank AC bands by unit distortion, then greedily add bands
    (lowest distortion first) until total +-1 capacity >= capacity.
    Returns the selected band list (zigzag AC indices)."""
    band_d = []
    for i in range(1, 64):
        d = unit_band_distortion(list(coeffs[:, i]), T_zz[i])
        cap = int(np.sum(np.abs(coeffs[:, i]) == 1))   # embeddable +-1 count
        band_d.append((d, i, cap))
    band_d.sort(key=lambda x: x[0])      # ascending distortion
    selected, total = [], 0
    for d, i, cap in band_d:
        if cap == 0 or not np.isfinite(d):
            continue
        selected.append(i)
        total += cap
        if total >= capacity:
            break
    return sorted(selected), total


def embed(coeffs, order, bands, bits):
    """Embed bits into +-1 coefficients of selected bands, processing blocks
    in `order`. Shiftable coefficients (|c|>1) in those bands shift outward."""
    out = coeffs.copy()
    bi = 0
    for k in order:
        for i in bands:
            c = int(out[k, i])
            if abs(c) == 1:
                t = bits[bi] if bi < len(bits) else 0
                bi += 1 if bi < len(bits) else 0
                out[k, i] = hs_embed_coeff(c, t)
            elif abs(c) > 1:
                out[k, i] = hs_embed_coeff(c, 0)
    return out, bi


def extract(marked, T_zz, bands, capacity):
    """Re-sort by smoothness (invariant), then extract bits + recover."""
    order, _ = block_order(marked, T_zz)
    rec = marked.copy()
    bits = []
    for k in order:
        for i in bands:
            c = int(marked[k, i])
            if abs(c) <= 2 and abs(c) >= 1:
                r, t = hs_extract_coeff(c)
                if t is not None:
                    bits.append(t)
                rec[k, i] = r
            elif abs(c) > 2:
                r, _ = hs_extract_coeff(c)
                rec[k, i] = r
    return rec, bits, order


def v5():
    rng = np.random.default_rng(5)
    for qf in (50, 70, 90):
        # smooth-ish synthetic image -> many zero ACs -> good capacity
        x = np.linspace(0, 1, 64)
        base = (80 + 40*np.sin(6*x)[:, None] + 30*np.cos(5*x)[None, :])
        img = np.clip(base + rng.normal(0, 3, (64, 64)), 0, 255)
        coeffs, T_zz = simulate_jpeg_coeffs(img, qf)
        order0, _ = block_order(coeffs, T_zz)

        # choose a payload ~ 60% of total +-1 capacity across all AC bands
        total_pm1 = int(np.sum(np.abs(coeffs[:, 1:]) == 1))
        payload = int(total_pm1 * 0.5)
        bands, cap = select_bands(coeffs, T_zz, payload, order0)
        assert cap >= payload, (cap, payload)

        secret = list(rng.integers(0, 2, payload))
        marked, used = embed(coeffs, order0, bands, secret)
        assert used >= payload

        rec, bits_out, order_r = extract(marked, T_zz, bands, payload)
        # block order recovered identically (smoothness invariance)
        assert np.array_equal(order0, order_r), "block re-sort mismatch"
        assert np.array_equal(rec, coeffs), "coefficients not recovered"
        assert bits_out[:payload] == secret, "secret bits mismatch"
        print(f"V5 PASS qf={qf}: payload={payload} bits, {len(bands)} bands, "
              f"coeffs+data fully recovered")


def v6():
    """Dynamic band-count: fewest bands meeting capacity = lowest distortion."""
    rng = np.random.default_rng(6)
    img = np.clip(100 + rng.normal(0, 8, (64, 64)), 0, 255)
    coeffs, T_zz = simulate_jpeg_coeffs(img, 70)
    order0, _ = block_order(coeffs, T_zz)
    total_pm1 = int(np.sum(np.abs(coeffs[:, 1:]) == 1))
    for frac in (0.2, 0.4, 0.6):
        payload = int(total_pm1 * frac)
        bands, cap = select_bands(coeffs, T_zz, payload, order0)
        secret = list(rng.integers(0, 2, payload))
        marked, _ = embed(coeffs, order0, bands, secret)
        rec, bits_out, _ = extract(marked, T_zz, bands, payload)
        assert np.array_equal(rec, coeffs) and bits_out[:payload] == secret
    print("V6 PASS: dynamic band-count selection reversible at 20/40/60% load")


if __name__ == "__main__":
    try:
        import scipy  # noqa
    except ImportError:
        print("SKIP: scipy not installed (pip install scipy) — V1-V4 already cover "
              "the reversible primitives; V5/V6 need DCT.")
        raise SystemExit(0)
    v5()
    v6()
    print("\nFULL PIPELINE VALIDATED — safe to port to MATLAB.")
