"""Gerador de dados sintéticos para o Personalizado e o Entre-pessoas."""
import numpy as np

C, D = 12, 16
REP_TR, REP_TE = 4, 3
NTR, NTE = C * 2 * REP_TR, C * 2 * REP_TE
YTR = np.tile(np.repeat(np.arange(C), REP_TR), 2)
STR = np.repeat([0, 1], C * REP_TR)
YTE = np.tile(np.repeat(np.arange(C), REP_TE), 2)

MUNDOS = {
    "nulo_exato":        dict(tau=0.0, gamma=0.0, sigma0=0.9, h=0.0, omega=0.3, psi=0.3),
    "nulo_heterogeneo":  dict(tau=0.0, gamma=0.0, sigma0=0.9, h=0.5, omega=0.3, psi=0.3),
    "entre_baixa":       dict(tau=0.25, gamma=0.3, sigma0=0.9, h=0.3, omega=0.3, psi=0.3),
    "entre_media":       dict(tau=0.5, gamma=0.3, sigma0=0.9, h=0.3, omega=0.3, psi=0.3),
    "entre_alta":        dict(tau=1.0, gamma=0.3, sigma0=0.9, h=0.3, omega=0.3, psi=0.3),
    "pior_caso":         dict(tau=0.5, gamma=0.6, sigma0=0.9, h=0.7, omega=0.5, psi=0.5),
}
TAUS_VARREDURA = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def centros(seed):
    """Os 12 centros de frase, fixos por mundo (mesma semente em todos os N)."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (C, D))


def falantes(par, mu, N, rng):
    """Devolve Xtr (N,96,D), Xte (N,72,D)."""
    h = par["h"]
    z = rng.normal(size=(N, 2))
    sig = par["sigma0"] * np.exp(h * z[:, 0] - 0.5 * h * h)
    tau = par["tau"] * np.exp(h * z[:, 1] - 0.5 * h * h)
    delta = rng.normal(size=(N, C, D)) * tau[:, None, None]
    g = rng.normal(0.0, par["gamma"], (N, 1, D)) if par["gamma"] > 0 else np.zeros((N, 1, D))
    o = rng.normal(0.0, par["omega"], (N, 4, D))
    ps = rng.normal(0.0, par["psi"], (N, 4, C, D))
    base = mu[None] + delta + g
    Xtr = np.empty((N, NTR, D)); Xte = np.empty((N, NTE, D))
    for k in range(2):
        sl = slice(k * C * REP_TR, (k + 1) * C * REP_TR)
        m = np.repeat(base + o[:, k][:, None, :] + ps[:, k], REP_TR, axis=1)
        Xtr[:, sl] = m + rng.normal(size=m.shape) * sig[:, None, None]
        sl = slice(k * C * REP_TE, (k + 1) * C * REP_TE)
        m = np.repeat(base + o[:, 2 + k][:, None, :] + ps[:, 2 + k], REP_TE, axis=1)
        Xte[:, sl] = m + rng.normal(size=m.shape) * sig[:, None, None]
    return Xtr, Xte


def distancias(Xtr, Xte):
    N = Xtr.shape[0]
    A = Xte.reshape(-1, D); B = Xtr.reshape(-1, D)
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return d2.reshape(N, NTE, N, NTR)


def por_frase(acerto):
    """acerto (..., 72) booleano -> acerto por frase (..., 12)."""
    sh = acerto.shape[:-1]
    a = acerto.reshape(sh + (2, C, REP_TE))
    return a.mean(axis=(-3, -1))


def sorteio_igualado(N, K, rng, n_por_sessao=REP_TR):
    M = C * 2 * n_por_sessao
    J = np.empty((N, K, M), int); T = np.empty((N, K, M), int)
    tam = (N - 1) * REP_TR
    for i in range(N):
        outros = np.array([j for j in range(N) if j != i])
        u = rng.random((K, 2, C, tam))
        esc = np.argsort(u, axis=-1)[..., :n_por_sessao]
        jj = outros[esc // REP_TR]
        r = esc % REP_TR
        ses = np.arange(2)[None, :, None, None]
        cc = np.arange(C)[None, None, :, None]
        tt = ses * C * REP_TR + cc * REP_TR + r
        J[i] = jj.reshape(K, -1); T[i] = tt.reshape(K, -1)
    return J, T


def sorteio_proprio(K, rng, n_por_sessao):
    """Subamostra do próprio treino (curva de aprendizagem do Personalizado)."""
    u = rng.random((K, 2, C, REP_TR))
    esc = np.argsort(u, axis=-1)[..., :n_por_sessao]
    ses = np.arange(2)[None, :, None, None]; cc = np.arange(C)[None, None, :, None]
    return (ses * C * REP_TR + cc * REP_TR + esc).reshape(K, -1)


def avaliar_knn(d2, J=None, T=None):
    N = d2.shape[0]
    idx = d2.argmin(axis=3)
    val = np.take_along_axis(d2, idx[..., None], 3)[..., 0]
    lab = YTR[idx]
    hit = lab == YTE[None, :, None]
    M = por_frase(np.moveaxis(hit, 2, 1))
    v = val.copy()
    v[np.arange(N), :, np.arange(N)] = np.inf
    jbest = v.argmin(axis=2)
    labf = np.take_along_axis(lab, jbest[..., None], 2)[..., 0]
    Ef = por_frase(labf == YTE[None, :])
    Eg = None
    if J is not None:
        K = J.shape[1]
        Eg = np.empty((N, K, C))
        for i in range(N):
            sub = d2[i][:, J[i], T[i]]
            a = sub.argmin(axis=2)
            labg = YTR[np.take_along_axis(T[i], a.T, 1)]
            Eg[i] = por_frase(labg == YTE[None, :])
    return M, Ef, Eg


def avaliar_centroide(Xtr, Xte, J=None, T=None):
    """Centroide mais próximo (classificador linear simples), mesmas saídas."""
    N = Xtr.shape[0]
    cen = np.stack([Xtr[:, YTR == c].mean(1) for c in range(C)], 1)

    def prever(X, cn):
        d = ((X[:, None, :] - cn[None]) ** 2).sum(-1)
        return d.argmin(1)
    M = np.empty((N, N, C)); Ef = np.empty((N, C))
    for i in range(N):
        for j in range(N):
            M[i, j] = por_frase(prever(Xte[i], cen[j]) == YTE)
        cf = (cen.sum(0) - cen[i]) / (N - 1)
        Ef[i] = por_frase(prever(Xte[i], cf) == YTE)
    Eg = None
    if J is not None:
        K, Mn = J.shape[1], J.shape[2]
        n = Mn // (2 * C)
        Eg = np.empty((N, K, C))
        for i in range(N):
            Xs = Xtr[J[i], T[i]].reshape(K, 2, C, n, D)
            cn = Xs.mean(axis=(1, 3))
            d = ((Xte[i][None, :, None, :] - cn[:, None]) ** 2).sum(-1)
            Eg[i] = por_frase(d.argmin(-1) == YTE[None, :])
    return M, Ef, Eg


def ganhos(M, Ef, Eg=None):
    """Diferenças por frase (N,12) Personalizado - cada versão do Entre-pessoas."""
    N = M.shape[0]
    P = M[np.arange(N), np.arange(N)]
    off = (M.sum(1) - P) / (N - 1)
    out = {"P": P, "completo": P - Ef, "doador_unico": P - off}
    if Eg is not None:
        out["igualado"] = P - Eg.mean(1)
    return out

