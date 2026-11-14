"""
sismo_la_guaira_danio.py -- LandSense
Mapa de dano aproximado (Damage Proxy) del sismo de La Guaira con la COHERENCIA NISAR a 20 m.

Idea (tecnica conocida: los "Damage Proxy Maps" de JPL/ARIA usan perdida de coherencia):
  si un edificio se derrumba, la senal del radar en ese punto cambia por completo entre dos
  pasadas -> la coherencia cae. Se mide cuanto cae en el par que CRUZA el sismo comparado con
  lo NORMAL para ese punto (linea base de pares anteriores).

Rigor:
  - Linea base: 4 pares de 12 dias previos al sismo -> media y dispersion de la coherencia por pixel.
  - CONTROL incorporado: el par previo mas reciente se trata "como si fuera el sismo" contra los
    3 anteriores. El umbral se fija en el percentil 99 de ese control (falsa alarma ~1% por
    construccion), no a ojo.
  - Solo pixeles ESTABLES antes (coherencia media de base >= 0.4): en vegetacion la coherencia ya
    es baja y el metodo no informa (se reporta cuanta superficie queda sin informacion).
  - Mascara oficial del grupo wrappedInterferogram (agua / subswath invalido fuera).
  - Dos orbitas independientes (061 ascendente, 126 descendente): se mide su coincidencia.
  - La perdida de coherencia NO es solo derrumbe: tambien deslizamientos, obras, cambios de
    vegetacion o agua. Por eso se VALIDA despues contra dano observado en el terreno
    (puntos de campo de HOT: src/sismo_la_guaira_danio_validacion.py).

Uso:
    python src/sismo_la_guaira_danio.py
Salidas: ../data/processed/la_guaira/danio/ (danio.txt, danio.png, dpm_<track>.npz)
"""
print("Iniciando mapa de dano por coherencia... (cargar librerias puede tardar 1-2 min)", flush=True)
import os, re, time
from datetime import datetime
import numpy as np
import requests
import h5py
import earthaccess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
print("Librerias cargadas.", flush=True)

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "la_guaira", "danio")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "danio.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# Franja urbana costera Catia La Mar - Naiguata + norte de Caracas
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -67.12, -66.72, 10.45, 10.65
SISMO = "20260624"
TRACKS = ["061", "126"]
N_BASE = 4               # pares previos de 12 dias para la linea base
MIN_ESTABLE = 0.4        # coherencia media de base minima para que el pixel informe
PISO_SIGMA = 0.05        # dispersion minima (evita dividir por casi cero)
COLECCION = "C2854335566-ASF"
W = "science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"

# ---------- 1) Pares por track (CMR directo) ----------
url = ("https://cmr.earthdata.nasa.gov/search/granules.json"
       f"?collection_concept_id={COLECCION}&bounding_box={LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"
       "&temporal=2026-03-01T00:00:00Z,2026-07-10T00:00:00Z&page_size=200")
pares = {t: [] for t in TRACKS}
for e in requests.get(url, timeout=60).json()["feed"]["entry"]:
    m = re.search(r"GUNW_\d{3}_(\d{3})_[AD]_\d{3}_\d{3}_\d{4}_SH_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{8})T", e["title"])
    h5 = [l["href"] for l in e.get("links", []) if l["href"].startswith("https") and l["href"].endswith(".h5")]
    if not (m and h5) or m.group(1) not in TRACKS:
        continue
    ref, sec = m.group(2), m.group(3)
    dias = (datetime.strptime(sec, "%Y%m%d") - datetime.strptime(ref, "%Y%m%d")).days
    if dias == 12:                                    # solo pares de 12 dias (comparables entre si)
        pares[m.group(1)].append(dict(ref=ref, sec=sec, url=h5[0]))

plan = {}
for t in TRACKS:
    ps = sorted({(q["ref"], q["sec"]): q for q in pares[t]}.values(), key=lambda q: q["ref"])
    evento = [q for q in ps if q["ref"] <= SISMO <= q["sec"]]
    previos = [q for q in ps if q["sec"] < SISMO][-N_BASE:]
    assert len(evento) == 1, f"track {t}: se esperaba 1 par que cruza el sismo, hay {len(evento)}"
    assert len(previos) == N_BASE, f"track {t}: faltan pares previos ({len(previos)}/{N_BASE})"
    plan[t] = dict(evento=evento[0], base=previos)
    p(f"Track {t}: evento {evento[0]['ref']}->{evento[0]['sec']} | base: " +
      ", ".join(f"{q['ref']}->{q['sec']}" for q in previos))

# ---------- 2) Lectura por streaming (coherencia 20 m + mascara oficial) ----------
earthaccess.login(persist=True)

def mascara_oficial(mk):
    m8 = np.where(mk == 255, 0, mk & 0xFF)
    return (mk != 255) & ((m8 // 10) % 10 > 0) & (m8 % 10 > 0) & ((m8 // 100) != 1)

def leer(u):
    for intento in range(1, 5):
        try:
            with h5py.File(earthaccess.open([u])[0], "r") as f:
                x = f[f"{W}/xCoordinates"][:]; y = f[f"{W}/yCoordinates"][:]
                epsg = int(f[f"{W}/projection"][()])
                tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                xa, ya = tr.transform(LON_MIN, LAT_MIN); xb, yb = tr.transform(LON_MAX, LAT_MAX)
                c = np.where((x >= min(xa, xb)) & (x <= max(xa, xb)))[0]
                r = np.where((y >= min(ya, yb)) & (y <= max(ya, yb)))[0]
                r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
                co = f[f"{W}/HH/coherenceMagnitude"][r0:r1, c0:c1].astype(np.float32)
                mk = f[f"{W}/mask"][r0:r1, c0:c1].astype(np.int32)
            co[~mascara_oficial(mk)] = np.nan
            return co, x[c0:c1], y[r0:r1], epsg
        except Exception as ex:
            p(f"    reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})"); time.sleep(6)
    raise SystemExit(f"No se pudo leer {u}")

def dpm(base, evento):
    """Caida de coherencia del evento respecto de lo normal, en unidades de dispersion (z)."""
    B = np.stack(base)
    mu = np.nanmean(B, 0); sg = np.maximum(np.nanstd(B, 0), PISO_SIGMA)
    estable = np.isfinite(mu) & (mu >= MIN_ESTABLE) & np.isfinite(evento) & (np.isfinite(B).sum(0) == len(base))
    z = np.where(estable, (mu - evento) / sg, np.nan)
    return z, mu, estable

R = {}
for t in TRACKS:
    p(f"\n===== Track {t} =====")
    base, grid = [], None
    for q in plan[t]["base"] + [plan[t]["evento"]]:
        co, x, y, epsg = leer(q["url"])
        if grid is None:
            grid = (x, y, epsg)
        else:
            assert np.array_equal(x, grid[0]) and np.array_equal(y, grid[1]), "grilla distinta entre pares del mismo track"
        p(f"  {q['ref']}->{q['sec']}: coherencia mediana {np.nanmedian(co):.2f} | validos {np.isfinite(co).mean():.0%}")
        base.append(co)
    ev = base.pop()                          # el ultimo leido es el evento

    # CONTROL: el par previo mas reciente como "falso evento" contra los 3 anteriores
    z_ctrl, _, est_ctrl = dpm(base[:-1], base[-1])
    umbral = float(np.nanpercentile(z_ctrl[est_ctrl], 99))
    z, mu, estable = dpm(base, ev)
    marcado = estable & (z >= umbral)
    p(f"  Pixeles estables (informan): {estable.mean():.0%} del recorte | sin informacion (vegetacion/agua/mascara): {1 - estable.mean():.0%}")
    p(f"  Umbral por control (p99 del falso evento): z >= {umbral:.2f}")
    p(f"  CONTROL: marcados {np.mean(z_ctrl[est_ctrl] >= umbral):.1%} (por construccion ~1%)")
    p(f"  EVENTO: marcados {marcado.sum() / max(estable.sum(), 1):.1%} de los pixeles estables "
      f"(= {marcado.sum() * 400 / 1e6:.2f} km2)")
    np.savez_compressed(os.path.join(OUT, f"dpm_{t}.npz"), z=z.astype(np.float32), mu=mu.astype(np.float32),
                        ev=ev.astype(np.float32), marcado=marcado, estable=estable, umbral=umbral,
                        z_ctrl=z_ctrl.astype(np.float32), x=grid[0], y=grid[1], epsg=grid[2])
    R[t] = dict(z=z, z_ctrl=z_ctrl, mu=mu, marcado=marcado, estable=estable, umbral=umbral,
                x=grid[0], y=grid[1], epsg=grid[2])

# ---------- 3) Coincidencia entre orbitas (061 ascendente vs 126 descendente) ----------
A, D = R["061"], R["126"]
XA, YA = np.meshgrid(A["x"], A["y"])
XD, YD = Transformer.from_crs(f"EPSG:{A['epsg']}", f"EPSG:{D['epsg']}", always_xy=True).transform(XA, YA)
ix = np.round((XD - D["x"][0]) / (D["x"][1] - D["x"][0])).astype(int)
iy = np.round((YD - D["y"][0]) / (D["y"][1] - D["y"][0])).astype(int)
ok = (ix >= 0) & (ix < len(D["x"])) & (iy >= 0) & (iy < len(D["y"]))
m_D = np.zeros_like(A["marcado"]); e_D = np.zeros_like(A["estable"])
m_D[ok] = D["marcado"][iy[ok], ix[ok]]; e_D[ok] = D["estable"][iy[ok], ix[ok]]
ambos = A["estable"] & e_D
if ambos.sum() > 1000:
    mA, mD = A["marcado"] & ambos, m_D & ambos
    p_azar = mA.mean() / max(ambos.mean(), 1e-9) * mD.mean() / max(ambos.mean(), 1e-9)
    coinc = (mA & mD).sum() / max(mA.sum(), 1)
    p(f"\nCoincidencia entre orbitas (pixeles estables en ambas: {ambos.sum()}):")
    p(f"  de lo que marca 061, {coinc:.0%} lo marca tambien 126 (al azar se esperaria ~{mD.sum() / ambos.sum():.0%})")
else:
    p("\nPoca superficie estable en comun entre orbitas: no se evalua coincidencia")

# ---------- 4) Figura ----------
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
for j, t in enumerate(TRACKS):
    r = R[t]; ext = [r["x"][0], r["x"][-1], r["y"][-1], r["y"][0]]
    ax[0, j].imshow(np.where(r["estable"], r["mu"], np.nan), cmap="gray", vmin=0, vmax=1, extent=ext)
    ax[0, j].imshow(np.where(r["marcado"], 1.0, np.nan), cmap="Reds", vmin=0, vmax=1, extent=ext)
    ax[0, j].set_title(f"Track {t} EVENTO: rojo = caida de coherencia anormal (posible dano)\n"
                       f"gris = coherencia normal (terreno estable)", fontsize=9)
    ax[1, j].imshow(np.where(r["estable"], r["mu"], np.nan), cmap="gray", vmin=0, vmax=1, extent=ext)
    ax[1, j].imshow(np.where(r["estable"] & (r["z_ctrl"] >= r["umbral"]), 1.0, np.nan), cmap="Reds",
                    vmin=0, vmax=1, extent=ext)
    ax[1, j].set_title(f"Track {t} CONTROL (par previo como falso evento): lo marcado es falsa alarma", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(OUT, "danio.png"), dpi=100)
p(f"\nFigura: {os.path.join(OUT, 'danio.png')}")
p("Siguiente: validar contra dano observado en el terreno (src/sismo_la_guaira_danio_validacion.py).")
LOG.close()
