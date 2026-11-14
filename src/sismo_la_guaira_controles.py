"""
sismo_la_guaira_controles.py -- LandSense
PRUEBA DE CONTROL del co-sismico La Guaira (24-jun-2026).

Hallazgo previo (src/sismo_la_guaira_auditoria.py): la "franja roja" de -46..-69 cm
era un ARTEFACTO: ~46k pixeles con fase desenrollada EXACTAMENTE 0 (zona no
desenrollada, rellena con 0) menos la pantalla ionosferica (~24.6 rad). Se excluyen.

Pregunta de este script: el gradiente N-S que queda (~50 cm), es el SISMO o es
ionosfera/geometria? Prueba: mismo track (162 ascendente), 3 pares:
    01->13 jun  (ANTES, control: no deberia tener senal sismica)
    13->25 jun  (CRUZA el sismo)
    25 jun->07 jul (DESPUES, control: a lo sumo post-sismico chico)
Si el patron aparece SOLO en el par que cruza -> es el sismo.

Uso:
    python src/sismo_la_guaira_controles.py
Salidas: ../data/processed/la_guaira/control.png, control.txt, gunw_<ref>_<sec>.npz
"""
print("Iniciando control La Guaira... (cargar librerias puede tardar 1-2 min)", flush=True)
import os, re, time, json
import numpy as np
import h5py
import earthaccess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
print("Librerias cargadas. Login Earthdata...", flush=True)

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "la_guaira")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "control.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

LON_MIN, LON_MAX = -67.60, -66.50
LAT_MIN, LAT_MAX = 10.30, 10.80
SISMO = "20260624"
TRACK = "_162_A_"
BASE = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
FREQ = "science/LSAR/GUNW/grids/frequencyA/centerFrequency"
C_LUZ = 299792458.0

earthaccess.login(persist=True)
res = earthaccess.search_data(short_name="NISAR_L2_GUNW_PROVISIONAL_V1",
                              bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
                              temporal=("2026-05-25", "2026-07-15"))
pares = []
for g in res:
    links = g.data_links()
    name = links[0].split("/")[-1] if links else str(g)
    d = re.findall(r"(\d{8})T\d{6}", name)
    if TRACK in name and len(d) >= 3:
        pares.append((g, name, d[0], d[2]))
pares.sort(key=lambda t: t[2])
p(f"Pares del track {TRACK.strip('_')}: {len(pares)}")
for _, n, r, s in pares:
    tipo = "CRUZA SISMO" if r <= SISMO <= s else ("ANTES" if s < SISMO else "DESPUES")
    p(f"  {r} -> {s}  [{tipo}]")

def leer(fobj):
    with h5py.File(fobj, "r") as f:
        lam = C_LUZ / float(f[FREQ][()])
        x = f[f"{BASE}/xCoordinates"][:]; y = f[f"{BASE}/yCoordinates"][:]
        epsg = int(f[f"{BASE}/projection"][()])
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xa, ya = tr.transform(LON_MIN, LAT_MIN); xb, yb = tr.transform(LON_MAX, LAT_MAX)
        c = np.where((x >= min(xa, xb)) & (x <= max(xa, xb)))[0]
        r = np.where((y >= min(ya, yb)) & (y <= max(ya, yb)))[0]
        r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
        return dict(
            ph=f[f"{BASE}/unwrappedPhase"][r0:r1, c0:c1].astype(np.float32),
            io=f[f"{BASE}/ionospherePhaseScreen"][r0:r1, c0:c1].astype(np.float32),
            co=f[f"{BASE}/coherenceMagnitude"][r0:r1, c0:c1].astype(np.float32),
            cc=f[f"{BASE}/connectedComponents"][r0:r1, c0:c1].astype(np.int32),
            x=x[c0:c1], y=y[r0:r1], epsg=epsg, lam=lam)

def plano(los, x, y):
    """Ajusta a + b*x + c*y; devuelve gradiente E-O y N-S en cm por 10 km."""
    X, Y = np.meshgrid(x, y)
    m = np.isfinite(los)
    A = np.c_[np.ones(m.sum()), (X[m] - X[m].mean()) / 1e4, (Y[m] - Y[m].mean()) / 1e4]
    coef, *_ = np.linalg.lstsq(A, los[m], rcond=None)
    return coef[1], coef[2]

filas = []
for g, name, ref, sec in pares:
    tipo = "CRUZA SISMO" if ref <= SISMO <= sec else ("ANTES" if sec < SISMO else "DESPUES")
    p(f"\n===== {ref} -> {sec} [{tipo}] =====")
    for intento in range(1, 5):
        try:
            d = leer(earthaccess.open([g])[0]); break
        except Exception as e:
            p(f"  reintento {intento} ({type(e).__name__})"); time.sleep(6)
    else:
        p("  no se pudo leer"); continue
    p(f"  lambda = {d['lam']:.4f} m (de la frecuencia del producto)")
    base = np.isfinite(d["ph"]) & (d["co"] >= 0.2) & (d["cc"] > 0)
    cero = base & (d["ph"] == 0)
    ok = base & ~cero
    p(f"  validos {int(base.sum())} | con fase EXACTAMENTE 0 (excluidos): {int(cero.sum())} | usados {int(ok.sum())}")
    k = d["lam"] / (4 * np.pi) * 100
    sin_io = np.where(ok, d["ph"] * k, np.nan)
    con_io = np.where(ok & np.isfinite(d["io"]), (d["ph"] - d["io"]) * k, np.nan)
    salida = {}
    for nom, los in [("SIN ionosfera", sin_io), ("CON ionosfera", con_io)]:
        los = los - np.nanmedian(los)
        ge, gn = plano(los, d["x"], d["y"])
        rango = np.nanpercentile(los, 99) - np.nanpercentile(los, 1)
        p(f"  {nom}: rango p1-p99 {rango:5.1f} cm | gradiente E-O {ge:+5.2f} cm/10km | N-S {gn:+5.2f} cm/10km")
        salida[nom] = los
    np.savez_compressed(os.path.join(OUT, f"gunw_{ref}_{sec}.npz"),
                        **{k2: v for k2, v in d.items()})
    filas.append((ref, sec, tipo, salida, d))

n = len(filas)
fig, ax = plt.subplots(n, 2, figsize=(14, 4.5 * n), squeeze=False)
for i, (ref, sec, tipo, salida, d) in enumerate(filas):
    ext = [d["x"][0], d["x"][-1], d["y"][-1], d["y"][0]]
    for j, nom in enumerate(["SIN ionosfera", "CON ionosfera"]):
        im = ax[i, j].imshow(salida[nom], cmap="RdBu", vmin=-30, vmax=30, extent=ext, origin="upper")
        ax[i, j].set_title(f"{ref}->{sec} [{tipo}] {nom} (cm, rel. a mediana)")
        plt.colorbar(im, ax=ax[i, j])
plt.tight_layout()
fig.savefig(os.path.join(OUT, "control.png"), dpi=100)
p(f"\nFigura: {os.path.join(OUT, 'control.png')}")
p("Listo.")
LOG.close()
