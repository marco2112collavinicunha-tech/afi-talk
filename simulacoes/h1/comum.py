"""Funções comuns das simulações da primeira comparação (dados sintéticos)."""
import os, sys, hashlib, platform
import numpy as np, scipy
from scipy import stats
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ); sys.path.insert(0, os.path.join(RAIZ, "analise"))
import analise_do_estudo as prog
SEMENTE = 20261004
ARQ_PROG = os.path.join(RAIZ, "analise", "analise_do_estudo.py")


def meta():
    return {"SINTETICO": True, "semente_raiz": SEMENTE, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "programa": "analise/analise_do_estudo.py",
            "sha256_programa": hashlib.sha256(open(ARQ_PROG, "rb").read()).hexdigest(),
            "aviso": "Dados SINTÉTICOS; não cobre fala apráxica real nem o tamanho real da assimetria, "
                     "do agrupamento ou das trocas."}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def logistica(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


ALFA = prog.ALFA_DA_H1A   # nível do estudo (4%); cada teste aceita outro nível


def t_johnson(d, alfa=ALFA):
    n = d.shape[1]
    m = d.mean(1); s = d.std(1, ddof=1)
    mu3 = ((d - m[:, None]) ** 3).mean(1)
    ok = s > 1e-9
    S = np.where(ok, m / np.where(ok, s, 1), 0)
    g = np.where(ok, mu3 / np.where(ok, s, 1) ** 3, 0)
    tj = np.sqrt(n) * (S + g / (6 * n) + g / 3 * S ** 2)
    return ok & (tj > stats.t.ppf(1 - alfa, n - 1))


def bootstrap_t(d, rng, B=999, alfa=ALFA):
    R, n = d.shape
    m = d.mean(1); s = d.std(1, ddof=1)
    ok = s > 1e-9
    tobs = np.where(ok, m / np.where(ok, s, 1) * np.sqrt(n), 0)
    rej = np.zeros(R, bool)
    passo = max(1, 400000 // (B * n))
    for a in range(0, R, passo):
        b = min(R, a + passo)
        idx = rng.integers(0, n, size=(b - a, B, n))
        x = np.take_along_axis(np.broadcast_to(d[a:b, None, :], (b - a, B, n)), idx, axis=2)
        ms = x.mean(2); ss = x.std(2, ddof=1)
        dif = ms - m[a:b, None]
        z = ss > 1e-12
        with np.errstate(invalid='ignore', divide='ignore'):
            ts = np.where(z, dif / np.where(z, ss, 1) * np.sqrt(n),
                          np.where(np.abs(dif) <= 1e-12, 0.0, np.sign(dif) * np.inf))
        p = (1 + (ts >= tobs[a:b, None] - 1e-12).sum(1)) / (B + 1)
        rej[a:b] = ok[a:b] & (p <= alfa)
    return rej


_SINAIS = {}


def _sinais01(n):
    if n not in _SINAIS:
        _SINAIS[n] = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    return _SINAIS[n]


def wilcoxon_exato(d, alfa=ALFA):
    R, n = d.shape
    a = np.abs(d)
    zero = a < 1e-12
    nz = zero.sum(1)
    r = stats.rankdata(np.where(zero, -1.0, a), axis=1, method="average") - nz[:, None]
    r = np.where(zero, 0.0, r)
    wobs = (r * (d > 0)).sum(1)
    perm = _sinais01(n) @ r.T
    p = (perm >= wobs[None, :] - 1e-9).mean(0)
    return (nz < n) & (p <= alfa)


def aparada(d, versao, alfa=ALFA):
    R, n = d.shape
    g = int(np.floor(0.2 * n)); h = n - 2 * g
    x = np.sort(d, 1)
    tm = x[:, g:n - g].mean(1)
    w = np.clip(x, x[:, [g]], x[:, [n - g - 1]])
    if versao == "a":
        se = w.std(1, ddof=1) / ((1 - 0.4) * np.sqrt(n)); gl = n - 2 * g - 1
    else:
        sqd = ((w - w.mean(1, keepdims=True)) ** 2).sum(1)
        se = np.sqrt(sqd / (h * (h - 1))); gl = h - 1
    ok = se > 1e-12
    T = np.where(ok, tm / np.where(ok, se, 1), 0)
    return ok & (T > stats.t.ppf(1 - alfa, gl))


def t_por_grupo(d, grupos, alfa=ALFA):
    """t de uma amostra sobre as médias por grupo (grupos: vetor de rótulos de comprimento N)."""
    rot = np.unique(grupos)
    G = len(rot)
    if G < 2:
        return np.zeros(d.shape[0], bool)
    mg = np.stack([d[:, grupos == k].mean(1) for k in rot], 1)
    m = mg.mean(1); s = mg.std(1, ddof=1)
    ok = s > 1e-9
    T = np.where(ok, m / np.where(ok, s, 1) * np.sqrt(G), 0)
    return ok & (T > stats.t.ppf(1 - alfa, G - 1))

