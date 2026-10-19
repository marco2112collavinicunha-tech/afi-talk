"""Funções comuns das simulações do Entre-pessoas (dados sintéticos)."""
import os, sys, hashlib, platform, itertools
import numpy as np, scipy
from scipy import stats

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ); sys.path.insert(0, os.path.join(RAIZ, "analise"))
import analise_do_estudo as prog

SEMENTE = 20261004
ALFA = 0.05


def sha_programa():
    return hashlib.sha256(open(os.path.join(RAIZ, "analise", "analise_do_estudo.py"),
                               "rb").read()).hexdigest()


def meta():
    return {"SINTETICO": True, "semente_raiz": SEMENTE, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "programa": "analise/analise_do_estudo.py",
            "sha256_programa": sha_programa(),
            "funcoes_do_programa": ["ganho_frases_fixas", "teste_nadeau_bengio_leave_one_out"],
            "aviso": "Dados SINTÉTICOS (vetores gaussianos, kNN-1/centroide); não é fala apráxica real."}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def t_simples(Dpf):
    """t do programa (degraus 1 e 2) sobre a matriz participante x frase."""
    r = prog.ganho_frases_fixas(Dpf, alfa=ALFA)
    return (r["status"] == "passou"), r.get("ep")


def t_nb(g):
    r = prog.teste_nadeau_bengio_leave_one_out(list(g), alfa=ALFA)
    return bool(r["status"] == "calculado" and r["ganho_medio"] > 0 and r["p"] <= ALFA)


_PERMS = {}


def permutacoes(N, rng, nmc=4999):
    """Todas as N! se N <= 8; senão identidade + nmc sorteadas."""
    if N <= 8:
        if N not in _PERMS:
            _PERMS[N] = np.array(list(itertools.permutations(range(N))))
        return _PERMS[N]
    P = np.argsort(rng.random((nmc, N)), axis=1)
    return np.vstack([np.arange(N)[None], P])


def p_permutacao(A, perms):
    """A (N,N): acerto global do modelo de j testado em i."""
    N = A.shape[0]
    T = A[np.arange(N)[None, :], perms].sum(1)
    t0 = np.trace(A)
    return float(np.mean(T >= t0 - 1e-12))

