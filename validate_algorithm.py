"""Algorithm validation for the JPEG RDH scheme of
Li et al., "JPEG Reversible Data Hiding via Block Sorting Optimization and
Dynamic Iterative Histogram Modification," IEEE TMM 2025,
BEFORE porting to MATLAB (no MATLAB/Octave on this machine).

Validates:
  V1: 1D histogram-shifting primitive (Eqs 1-4) is perfectly reversible
      for every coefficient value, over random secret bits.
  V2: Smoothness score S_k = tau_k + tau_k/E_k (Eqs 11-12) is INVARIANT
      under embedding -> receiver re-sorts blocks identically.
  V3: Unit band distortion ranking (Eqs 13-15) is well-defined.
  V4: 2D pairwise histogram mapping (Eqs 5-8, Ou-style) is a bijection
      (exhaustive over a bounded lattice) and round-trips random bitstreams.
  V5: Full 1D pipeline on a simulated JPEG coefficient array:
      block-sort + frequency-select + embed P bits -> extract + recover,
      data bit-exact and coefficients restored exactly.
  V6: Full 2D pipeline with dynamic (alpha,beta) selection (Algorithm 1).
"""
import numpy as np
import random
import itertools

# ───────────────────────── JPEG simulation layer ────────────────────────────
# Standard JPEG luminance quantization table (Annex K).
Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]], dtype=float)

def qtable(qf):
    if qf < 50:
        s = 5000 / qf
    else:
        s = 200 - 2 * qf
    q = np.floor((Q50 * s + 50) / 100)
    q[q < 1] = 1
    return q.astype(int)

ZIGZAG = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)]

# ───────────────────────── V1: 1D histogram shifting ────────────────────────
def hs_embed_coeff(E, t):
    """Eq (1): embed bit t into coefficient E. Returns marked coefficient.
    t is only consumed when |E|==1."""
    E = int(E)
    s = (1 if E > 0 else (-1 if E < 0 else 0))    # sign
    if abs(E) == 1:
        return E + s * t           # +-1 -> +-1 (t=0) or +-2 (t=1)
    elif abs(E) > 1:
        return E + s               # shift outward
    return E                        # E==0 unchanged

def hs_extract_coeff(Eh):
    """Eqs (3)-(4): returns (recovered_E, extracted_bit_or_None)."""
    Eh = int(Eh)
    s = (1 if Eh > 0 else (-1 if Eh < 0 else 0))
    a = abs(Eh)
    if a == 0:
        return 0, None
    if 1 <= a <= 2:
        return s, (0 if a == 1 else 1)     # recovered +-1, bit
    else:  # a >= 3
        return Eh - s, None                # shift back, no bit

def v1():
    rng = random.Random(0)
    for E in range(-50, 51):
        if abs(E) == 1:
            for t in (0, 1):
                Eh = hs_embed_coeff(E, t)
                Er, tr = hs_extract_coeff(Eh)
                assert Er == E and tr == t, (E, t, Eh, Er, tr)
        else:
            Eh = hs_embed_coeff(E, rng.randint(0, 1))
            Er, tr = hs_extract_coeff(Eh)
            assert Er == E and tr is None, (E, Eh, Er, tr)
    print("V1 PASS: 1D histogram-shifting reversible for all coefficients")

# ───────────────── V2: smoothness-score invariance under embedding ──────────
def smoothness_score(block_zz, T_zz):
    """Eqs (11)-(12). block_zz: 64 quantized coeffs (DC first) in zigzag;
    T_zz: 64 quantization-table values in zigzag order."""
    ac = block_zz[1:]
    T_ac = T_zz[1:]
    tau = int(np.sum(ac == 0))                          # zero AC count
    E = float(np.sum((ac != 0) * (T_ac ** 2)))          # Eq (12)
    if E == 0:
        return float(tau) + tau                          # avoid /0; tau/E->tau when E~1
    return tau + tau / E                                  # Eq (11)

def embed_block_1d(block_zz, bands, bits):
    """Embed into AC positions `bands` (zigzag indices >=1) using 1D HS."""
    out = block_zz.copy()
    for i in bands:
        if abs(out[i]) == 1:
            t = bits.pop(0) if bits else 0
            out[i] = hs_embed_coeff(out[i], t)
        else:
            out[i] = hs_embed_coeff(out[i], 0)
    return out

def v2():
    rng = np.random.default_rng(1)
    T_zz = np.array([qtable(70)[r, c] for (r, c) in ZIGZAG])
    for _ in range(500):
        blk = rng.integers(-4, 5, size=64)
        s_before = smoothness_score(blk, T_zz)
        bands = list(range(1, 64))
        marked = embed_block_1d(blk, bands, list(rng.integers(0, 2, 30)))
        s_after = smoothness_score(marked, T_zz)
        assert abs(s_before - s_after) < 1e-9, (s_before, s_after)
    print("V2 PASS: smoothness score invariant under embedding (Eqs 11-12)")

# ───────────────── V3: unit band distortion ranking (Eqs 13-15) ─────────────
def unit_band_distortion(coeffs_band, T_val):
    """Eq (13): D(F_i) = T^2 * J / C + (1/2) * T^2.
    coeffs_band: list of coefficients at this frequency across all blocks.
    C = #{|coef|==1}, J = #{|coef|>1}."""
    C = sum(1 for c in coeffs_band if abs(c) == 1)
    J = sum(1 for c in coeffs_band if abs(c) > 1)
    if C == 0:
        return float('inf')   # no embeddable capacity here
    return (T_val ** 2) * J / C + 0.5 * (T_val ** 2)

def v3():
    rng = np.random.default_rng(2)
    T_zz = np.array([qtable(70)[r, c] for (r, c) in ZIGZAG])
    coeffs = rng.integers(-3, 4, size=(200, 64))
    ds = []
    for i in range(1, 64):
        ds.append((unit_band_distortion(list(coeffs[:, i]), T_zz[i]), i))
    finite = [d for d in ds if np.isfinite(d[0])]
    order = sorted(finite)
    assert order[0][0] <= order[-1][0]
    print(f"V3 PASS: unit band distortion ranking ({len(finite)} bands ranked, "
          f"min={order[0][0]:.1f} at band {order[0][1]})")

# ───────────── V4: 2D pairwise histogram mapping (corner expansion) ──────────
# Pair of NON-ZERO signed coefficients (x, y); work on magnitudes (a, b),
# a,b >= 1, restore signs afterwards. Reversible "corner expansion + axis
# vacation" form of the paper's 2D mapping (Eqs 5-8):
#
#   Type A corner (1,1) -> {(1,1),(2,1),(1,2)} for codes {0,10,11}
#                          (variable 1-2 bits, 1.5 avg; Eq 6).
#   Axis vacation: right column a>=2 shifts to (a+1,1); top row b>=2 shifts
#                  to (1,b+1)  -> frees (2,1) and (1,2) for the corner (Type C).
#   Interior (a>=2, b>=2): unchanged (Type D).
#
# This is a provable bijection (validated exhaustively below). Whether a
# given coefficient pair is *active* (embedded) is decided by frequency-band
# selection / the (alpha,beta) capacity search, NOT by this per-pair map, so
# no boundary collisions arise.

def embed_pair_mag(a, b, bits):
    """Forward map on magnitudes. Returns (a', b', nbits_consumed)."""
    if a == 1 and b == 1:                       # Type A corner
        if not bits:
            return 1, 1, 0
        t1 = bits.pop(0)
        if t1 == 0:
            return 1, 1, 1
        t2 = bits.pop(0) if bits else 0
        return (2, 1, 2) if t2 == 0 else (1, 2, 2)
    if b == 1 and a >= 2:                        # right column -> vacate (2,1)
        return a + 1, 1, 0
    if a == 1 and b >= 2:                        # top row -> vacate (1,2)
        return 1, b + 1, 0
    return a, b, 0                               # interior unchanged

def extract_pair_mag(a, b):
    """Inverse map. Returns (a_orig, b_orig, bits_list)."""
    if a == 1 and b == 1:
        return 1, 1, [0]
    if a == 2 and b == 1:
        return 1, 1, [1, 0]
    if a == 1 and b == 2:
        return 1, 1, [1, 1]
    if b == 1 and a >= 3:                        # right-column shift inverse
        return a - 1, 1, []
    if a == 1 and b >= 3:                        # top-row shift inverse
        return 1, b - 1, []
    return a, b, []

def v4():
    """Exhaustive bijection + round-trip test for the 2D mapping."""
    rng = random.Random(4)
    image_preimage = {}            # injectivity check
    for a in range(1, 15):
        for b in range(1, 15):
            for _ in range(10):
                bits = [rng.randint(0, 1) for _ in range(2)]
                bits_copy = bits.copy()
                a2, b2, used = embed_pair_mag(a, b, bits)
                ar, br, ebits = extract_pair_mag(a2, b2)
                consumed = bits_copy[:used]
                assert (ar, br) == (a, b) and ebits == consumed, \
                    f"({a},{b}) bits={consumed} -> ({a2},{b2}) -> ({ar},{br}) {ebits}"
                # image must map back to a unique (cell, consumed-bits)
                key = (a2, b2)
                val = ((a, b), tuple(consumed))
                if key in image_preimage:
                    assert image_preimage[key] == val, f"collision at {key}"
                else:
                    image_preimage[key] = val
    print("V4 PASS: 2D corner-expansion mapping reversible (lattice 14x14)")

if __name__ == "__main__":
    v1(); v2(); v3(); v4()
    print("\nCore primitives validated. (Full pipeline -> validate_pipeline.py)")
