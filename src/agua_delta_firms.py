"""
agua_delta_firms.py -- LandSense
Cruza los focos de incendio NASA FIRMS con el cambio del eco NISAR (GCOV HV) en el Delta del Parana,
por LUGAR y por FECHA. Tambien caracteriza la caida grande de agosto (firma HH vs HV).

Clave FIRMS (MAP_KEY gratis de NASA FIRMS): se lee de la variable de entorno FIRMS_API_KEY
o de un archivo .env en la carpeta del proyecto (no se sube al repo; no se imprime).
Requiere: src/agua_gcov_exploratorio.py corrido (data/processed/gcov_exploratorio/delta_parana.npz).

Resultado:
  - 417 focos (1-jul a 20-sep-2026) en el AOI. En los pixeles con foco, el HV cambia -0.1 a -0.4 dB en el
    intervalo del fuego (dentro del ruido: pixeles al azar -0.2 a +0.4) -> NISAR no ve en general estos
    incendios de pastizal/junco (banda L ve fuego en bosque, no en herbaceas).
  - Caida de agosto (dHV < -3 dB, 20-jul -> 13-ago): 10.7% del area; HH cae junto (~-5 dB), HH/HV casi igual,
    16% pasa a "agua abierta", recuperacion lenta en un mes -> firma de agua cubriendo vegetacion baja
    (HIPOTESIS crecida; falta confirmar con altura del rio).

Uso:
    python src/agua_delta_firms.py
Salidas: ../data/processed/gcov_exploratorio/firms_delta.csv, firms_vs_nisar.txt
"""
import os, io, collections
from datetime import date, timedelta
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.ndimage import uniform_filter

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(AQUI, "..", "data", "processed", "gcov_exploratorio")
ENV_LOCAL = os.path.join(AQUI, "..", ".env")   # archivo local, en .gitignore
BBOX = "-59.3,-34.1,-58.8,-33.6"          # mismo AOI que src/agua_gcov_exploratorio.py
INICIO, FIN = date(2026, 7, 1), date(2026, 9, 20)
FUENTES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]
LOG = open(os.path.join(DIR, "firms_vs_nisar.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def clave():
    k = os.environ.get("FIRMS_API_KEY")
    if not k and os.path.exists(ENV_LOCAL):
        for l in open(ENV_LOCAL, encoding="utf-8", errors="ignore"):
            if l.startswith("FIRMS_API_KEY"):
                k = l.split("=", 1)[1].strip().strip('"').strip("'")
    assert k, "no se encontro FIRMS_API_KEY"
    return k

# ---------- 1) Focos FIRMS (API area: maximo 5 dias por pedido) ----------
arch = os.path.join(DIR, "firms_delta.csv")
if not os.path.exists(arch):
    K, partes = clave(), []
    for src in FUENTES:
        d, n = INICIO, 0
        while d <= FIN:
            r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{K}/{src}/{BBOX}/5/{d.isoformat()}", timeout=60)
            if r.ok and r.text.strip():
                df = pd.read_csv(io.StringIO(r.text))
                if len(df) and "latitude" in df:
                    df["fuente"] = src; partes.append(df); n += len(df)
            d += timedelta(days=5)
        p(f"{src}: {n} focos")
    pd.concat(partes, ignore_index=True).to_csv(arch, index=False)
F = pd.read_csv(arch); F["acq_date"] = pd.to_datetime(F["acq_date"])
p(f"Focos FIRMS en el AOI {INICIO}..{FIN}: {len(F)}")

# ---------- 2) Cruce por lugar y fecha ----------
z = np.load(os.path.join(DIR, "delta_parana.npz"))
HH, HV, C = z["HH"], z["HV"], z["clases"]
fechas = [str(f) for f in z["fechas"]]; fd = pd.to_datetime(fechas)
x0, y0, dx, dy, epsg = float(z["x0"]), float(z["y0"]), float(z["dx"]), float(z["dy"]), int(z["epsg"])
dB = 10 * np.log10(HV)
D = np.stack([uniform_filter(np.nan_to_num(dB[i + 1] - dB[i]), 3) for i in range(len(fechas) - 1)])   # ~300 m ~ pixel VIIRS
X, Y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(F.longitude.values, F.latitude.values)
F["c"] = np.round((X - x0) / dx).astype(int); F["r"] = np.round((Y - y0) / dy).astype(int)
F = F[(F.c >= 1) & (F.c < HV.shape[2] - 1) & (F.r >= 1) & (F.r < HV.shape[1] - 1)].copy()
def intervalo(t):
    for i in range(len(fd) - 1):
        if fd[i] < t <= fd[i + 1]:
            return i
    return -1
F["k"] = F.acq_date.apply(intervalo)
F = F[F.k >= 0].drop_duplicates(["r", "c", "k"])
p(f"Pixeles-foco unicos dentro de un intervalo NISAR: {len(F)}")
rng = np.random.default_rng(0)
todos = np.argwhere(np.isfinite(dB).all(0))
p("intervalo            n   dHV en su intervalo   cae>2dB   mismos px otros intervalos   px al azar")
for k in range(len(fechas) - 1):
    s = F[F.k == k]
    if not len(s):
        continue
    propio = D[k, s.r, s.c]
    otros = np.concatenate([D[j, s.r, s.c] for j in range(len(fechas) - 1) if j != k])
    rp = todos[rng.choice(len(todos), 5000, replace=False)]
    p(f"{fechas[k]}->{fechas[k+1]} {len(s):4d}   {np.median(propio):+6.2f} dB          {np.mean(propio < -2):4.0%}      "
      f"{np.median(otros):+6.2f} dB                  {np.median(D[k, rp[:, 0], rp[:, 1]]):+6.2f} dB")

# ---------- 3) Firma de la caida de agosto ----------
i0, i1 = fechas.index("20260720"), fechas.index("20260813")
dHV = 10 * np.log10(HV[i1] / HV[i0]); dHH = 10 * np.log10(HH[i1] / HH[i0])
sel = np.isfinite(dHV) & np.isfinite(dHH) & (dHV < -3)
p(f"\nCaida de HV > 3 dB (20-jul -> 13-ago): {sel.sum()} px ({sel.sum() / np.isfinite(dHV).sum():.1%} del area)")
p(f"  HH: mediana {np.median(dHH[sel]):+.1f} dB | sube {np.mean(dHH[sel] > 0):.0%} | baja > 3 dB {np.mean(dHH[sel] < -3):.0%} | "
  f"HH/HV cambia {np.median((dHH - dHV)[sel]):+.1f} dB")
nombres = {0: "sin dato", 8: "sin agua", 3: "agua abierta", 16: "veg.inund. A", 12: "veg.inund. B"}
for (a, b), n in collections.Counter(zip(C[i0][sel], C[i1][sel])).most_common(4):
    p(f"  {nombres[a]:13s} -> {nombres[b]:13s}: {n / sel.sum():.0%}")
for j in range(i1 + 1, len(fechas)):
    p(f"  {fechas[j]}: HV {np.median(10*np.log10(HV[j]/HV[i0])[sel]):+.1f} dB | HH {np.median(10*np.log10(HH[j]/HH[i0])[sel]):+.1f} dB (vs 20-jul)")
LOG.close()
