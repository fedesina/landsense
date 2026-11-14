"""
sismo_la_guaira_descomposicion.py -- LandSense
Descompone el desplazamiento co-sismico de La Guaira en ESTE-OESTE y VERTICAL
combinando un par ascendente (track 061, 18->30 jun) y uno descendente
(track 126, 22 jun->4 jul), usando los vectores LOS REALES del producto.

Modelo por pixel (LOS positivo = hacia el satelite, convencion fijada en src/sismo_la_guaira_controles.py):
    d_asc  = eE_asc * E + eN_asc * N + eU_asc * U
    d_desc = eE_desc* E + eN_desc* N + eU_desc* U
Con 2 observaciones se resuelven 2 incognitas: SUPUESTO N = 0 (falla de rumbo E-O ->
movimiento mayormente E-O). El script reporta eN de cada track para dimensionar el error.

Controles:
  - La misma descomposicion sobre los pares de CONTROL (antes del sismo) debe dar ~0.
  - Fisica esperada (NASA): al sur de la falla, movimiento al OESTE (E negativo) de hasta
    ~60 cm, y |U| << |E| (falla de rumbo).
  - Referencia: zona lejana al sur (lat < LAT_REF) = 0. Todo es RELATIVO a esa zona
    (NASA tambien es relativo). Si esa zona se movio, los valores cambian: se reporta.

Requiere: haber corrido src/sismo_la_guaira_multiorbita.py (usa sus cos_*.npz).
Uso:
    python src/sismo_la_guaira_descomposicion.py
Salidas: ../data/processed/la_guaira/descomposicion.txt, descomposicion.png, EU_coseismico.npz
"""
print("Iniciando descomposicion E-O / vertical... (cargar librerias puede tardar 1-2 min)", flush=True)
import os, re, time
import numpy as np
import requests
import h5py
import earthaccess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
print("Librerias cargadas.", flush=True)

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "la_guaira")
LOG = open(os.path.join(OUT, "descomposicion.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -67.60, -66.50, 10.30, 10.80
AEROPUERTO = (-66.99, 10.60)
LAT_REF = 10.40            # zona de referencia: al sur de esta latitud (>= ~20 km de la costa)
ALTURA = 500.0             # m; nivel del cubo LOS a usar (el vector cambia muy poco con la altura)
ASC, DESC = "061", "126"
PARES = {"CRUZA": {ASC: ("20260618", "20260630"), DESC: ("20260622", "20260704")},
         "CONTROL": {ASC: ("20260606", "20260618"), DESC: ("20260610", "20260622")}}
RG = "science/LSAR/GUNW/metadata/radarGrid"
COLECCION = "C2854335566-ASF"

def cargar(track, ref, sec):
    f = os.path.join(OUT, f"cos_{track}_{ref}_{sec}.npz")
    assert os.path.exists(f), f"falta {f}: correr primero src/sismo_la_guaira_multiorbita.py"
    z = np.load(f)
    return dict(los=z["los_cm"].astype(np.float64), x=z["x"], y=z["y"], epsg=int(z["epsg"]))

def muestrear(campo, gx, gy, gepsg, X, Y, epsg_dest):
    """Vecino mas cercano del campo (grilla gx, gy en gepsg) en los puntos X, Y (epsg_dest)."""
    if gepsg != epsg_dest:
        X, Y = Transformer.from_crs(f"EPSG:{epsg_dest}", f"EPSG:{gepsg}", always_xy=True).transform(X, Y)
    ix = np.round((X - gx[0]) / (gx[1] - gx[0])).astype(int)
    iy = np.round((Y - gy[0]) / (gy[1] - gy[0])).astype(int)
    ok = (ix >= 0) & (ix < len(gx)) & (iy >= 0) & (iy < len(gy))
    out = np.full(X.shape, np.nan)
    out[ok] = campo[iy[ok], ix[ok]]
    return out

# ---------- 1) URLs de los pares co-sismicos (CMR directo) para leer los vectores LOS ----------
url = ("https://cmr.earthdata.nasa.gov/search/granules.json"
       f"?collection_concept_id={COLECCION}&bounding_box={LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"
       "&temporal=2026-06-10T00:00:00Z,2026-07-10T00:00:00Z&page_size=200")
urls = {}
for e in requests.get(url, timeout=60).json()["feed"]["entry"]:
    m = re.search(r"GUNW_\d{3}_(\d{3})_[AD]_\d{3}_\d{3}_\d{4}_SH_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{8})T", e["title"])
    h5 = [l["href"] for l in e.get("links", []) if l["href"].startswith("https") and l["href"].endswith(".h5")]
    if m and h5 and (m.group(2), m.group(3)) == PARES["CRUZA"].get(m.group(1)):
        urls[m.group(1)] = h5[0]
assert set(urls) == {ASC, DESC}, f"no se encontraron ambos pares en CMR: {list(urls)}"

earthaccess.login(persist=True)
def vectores(u):
    for intento in range(1, 5):
        try:
            with h5py.File(earthaccess.open([u])[0], "r") as f:
                h = f[f"{RG}/heightAboveEllipsoid"][:]; k = int(np.abs(h - ALTURA).argmin())
                v = dict(e=f[f"{RG}/losUnitVectorX"][k].astype(np.float64),
                         n=f[f"{RG}/losUnitVectorY"][k].astype(np.float64),
                         inc=f[f"{RG}/incidenceAngle"][k].astype(np.float64),
                         x=f[f"{RG}/xCoordinates"][:], y=f[f"{RG}/yCoordinates"][:],
                         epsg=int(f[f"{RG}/projection"][()]), h=float(h[k]))
            v["u"] = np.sqrt(np.clip(1 - v["e"] ** 2 - v["n"] ** 2, 0, 1))
            return v
        except Exception as ex:
            p(f"  reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})"); time.sleep(6)
    raise SystemExit("No se pudieron leer los vectores LOS")

V = {t: vectores(urls[t]) for t in (ASC, DESC)}
for t, v in V.items():
    p(f"Track {t}: nivel del cubo {v['h']:.0f} m | LOS este {np.nanmedian(v['e']):+.3f}, "
      f"norte {np.nanmedian(v['n']):+.3f}, arriba {np.nanmedian(v['u']):+.3f} | incidencia {np.nanmedian(v['inc']):.1f}")

# ---------- 2) Grilla comun (la del ascendente) ----------
def descomponer(etiqueta):
    A = cargar(ASC, *PARES[etiqueta][ASC]); D = cargar(DESC, *PARES[etiqueta][DESC])
    X, Y = np.meshgrid(A["x"], A["y"]); ep = A["epsg"]
    dA = A["los"]
    dD = muestrear(D["los"], D["x"], D["y"], D["epsg"], X, Y, ep)
    comp = {}
    for t in (ASC, DESC):
        for c in ("e", "n", "u"):
            comp[t + c] = muestrear(V[t][c], V[t]["x"], V[t]["y"], V[t]["epsg"], X, Y, ep)
    lon, lat = Transformer.from_crs(f"EPSG:{ep}", "EPSG:4326", always_xy=True).transform(X, Y)
    comun = np.isfinite(dA) & np.isfinite(dD) & np.isfinite(comp[ASC + "e"]) & np.isfinite(comp[DESC + "e"])
    ref = comun & (lat < LAT_REF)
    p(f"\n===== {etiqueta} =====")
    p(f"  pixeles en comun asc+desc: {int(comun.sum())} | en zona de referencia (lat<{LAT_REF}): {int(ref.sum())}")
    if ref.sum() < 2000:
        p("  zona de referencia chica: se usa la mediana de toda la zona comun (menos robusto)")
        ref = comun
    dA = dA - np.nanmedian(dA[ref]); dD = dD - np.nanmedian(dD[ref])
    eA, uA, eD, uD = comp[ASC + "e"], comp[ASC + "u"], comp[DESC + "e"], comp[DESC + "u"]
    det = eA * uD - eD * uA
    p(f"  determinante mediano {np.nanmedian(np.abs(det[comun])):.2f} (cerca de 0 = mal condicionado)")
    bien = comun & (np.abs(det) > 0.2)
    E = np.where(bien, (dA * uD - dD * uA) / det, np.nan)     # + = este
    U = np.where(bien, (eA * dD - eD * dA) / det, np.nan)     # + = arriba
    p(f"  |componente norte del LOS|: asc {np.nanmedian(np.abs(comp[ASC+'n'][comun])):.3f}, "
      f"desc {np.nanmedian(np.abs(comp[DESC+'n'][comun])):.3f} -> error por suponer N=0 ~ esa fraccion del mov. N-S")
    p(f"  E (este+): p1 {np.nanpercentile(E,1):+.1f} | mediana {np.nanmedian(E):+.1f} | p99 {np.nanpercentile(E,99):+.1f} cm")
    p(f"  U (arriba+): p1 {np.nanpercentile(U,1):+.1f} | mediana {np.nanmedian(U):+.1f} | p99 {np.nanpercentile(U,99):+.1f} cm")
    trA = Transformer.from_crs("EPSG:4326", f"EPSG:{ep}", always_xy=True)
    ax_, ay_ = trA.transform(*AEROPUERTO)
    ix = int(np.abs(A["x"] - ax_).argmin()); iy = int(np.abs(A["y"] - ay_).argmin())
    caja = (slice(iy, iy + 60), slice(max(0, ix - 30), ix + 30))   # ~5 km al sur del aeropuerto
    p(f"  Caja al sur del aeropuerto: validos {int(np.isfinite(E[caja]).sum())} | "
      f"E mediana {np.nanmedian(E[caja]):+.1f} cm | U mediana {np.nanmedian(U[caja]):+.1f} cm")
    return dict(E=E, U=U, x=A["x"], y=A["y"], epsg=ep, axp=ax_, ayp=ay_, lat=lat)

R = {k: descomponer(k) for k in ("CRUZA", "CONTROL")}

p("\nLECTURA (comparar con NASA: al sur de la falla, OESTE hasta ~60 cm, mayormente horizontal):")
p("  - CRUZA: E negativo grande al sur del aeropuerto y |U| mucho menor que |E| => coherente.")
p("  - CONTROL: E y U cerca de 0 (rango de pocos cm) => el metodo no inventa movimiento.")
p("  - Valores RELATIVOS a la zona de referencia; suponen N = 0.")

np.savez_compressed(os.path.join(OUT, "EU_coseismico.npz"), E_cm=R["CRUZA"]["E"].astype(np.float32),
                    U_cm=R["CRUZA"]["U"].astype(np.float32), x=R["CRUZA"]["x"], y=R["CRUZA"]["y"],
                    epsg=R["CRUZA"]["epsg"])

fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
r = R["CRUZA"]; ext = [r["x"][0], r["x"][-1], r["y"][-1], r["y"][0]]
for a, img, tit in [(ax[0], R["CRUZA"]["E"], "ESTE-OESTE co-sismico (cm)\nrojo = este, azul = oeste"),
                    (ax[1], R["CRUZA"]["U"], "VERTICAL co-sismico (cm)\nrojo = sube, azul = baja"),
                    (ax[2], R["CONTROL"]["E"], "CONTROL (antes del sismo): este-oeste (cm)\ndebe ser ~0")]:
    im = a.imshow(img, cmap="RdBu_r", vmin=-50, vmax=50, extent=ext, origin="upper")
    a.plot(r["axp"], r["ayp"], "k*", ms=12); a.set_title(tit, fontsize=10); plt.colorbar(im, ax=a)
plt.tight_layout(); fig.savefig(os.path.join(OUT, "descomposicion.png"), dpi=100)
p(f"\nFigura: {os.path.join(OUT, 'descomposicion.png')}")
p("Listo.")
LOG.close()
