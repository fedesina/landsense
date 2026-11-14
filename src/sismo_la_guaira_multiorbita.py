"""
sismo_la_guaira_multiorbita.py -- LandSense
La Guaira (24-jun-2026): TODOS los pares NISAR GUNW que cruzan el sismo + controles
+ validacion cruzada entre dos pares ascendentes independientes.

Por que existe: la busqueda con earthaccess devolvio 12 granulos y
la consulta directa al catalogo CMR devolvio 16; faltaban justo los pares
co-sismicos 18->30 jun (track 061, el que uso NASA, cubre el aeropuerto),
17->29 jun (054 descendente) y 22 jun->4 jul (126 descendente).
Aca se busca DIRECTO en CMR (publico) y se lee por streaming con earthaccess.

Criterios:
  - Mascara oficial del producto: unwrappedInterferogram/mask (agua / subswath ref /
    subswath sec; 0 = invalido), igual que MintPy. Ademas coherencia >= 0.2 y
    componentes conectados > 0. Se reporta cuantos pixeles con fase==0 sobreviven.
  - Ionosfera: se resta (reduce el ruido en los controles, ver src/sismo_la_guaira_controles.py).
  - SIGNO: LOS_hacia_satelite = -(fase - iono) * lambda / 4pi  (MintPy; coherente
    con NASA). Positivo = hacia el satelite. Colores como NASA: rojo = hacia, azul = lejos.
  - lambda desde centerFrequency del producto.
  - Control por track: el par inmediatamente anterior (no cruza el sismo).
  - Validacion cruzada: mapa 162 (13->25) vs 061 (18->30) en la zona que comparten.

Uso:
    python src/sismo_la_guaira_multiorbita.py
Salidas: ../data/processed/la_guaira/multitrack.txt, multitrack.png, cruzada.png,
         cos_<track>_<ref>_<sec>.npz
"""
print("Iniciando multitrack La Guaira... (cargar librerias puede tardar 1-2 min)", flush=True)
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
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "multitrack.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -67.60, -66.50, 10.30, 10.80
AEROPUERTO = (-66.99, 10.60)
SISMO = "20260624"
COLECCION = "C2854335566-ASF"          # NISAR_L2_GUNW_PROVISIONAL_V1
ROOT = "science/LSAR/GUNW/grids/frequencyA"
UNW = f"{ROOT}/unwrappedInterferogram"
C_LUZ = 299792458.0

# ---------- 1) Busqueda directa en CMR ----------
url = ("https://cmr.earthdata.nasa.gov/search/granules.json"
       f"?collection_concept_id={COLECCION}"
       f"&bounding_box={LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"
       "&temporal=2026-05-20T00:00:00Z,2026-07-20T00:00:00Z&page_size=200")
entradas = requests.get(url, timeout=60).json()["feed"]["entry"]
gran = []
for e in entradas:
    t = e["title"]
    m = re.search(r"GUNW_\d{3}_(\d{3})_([AD])_(\d{3})_\d{3}_\d{4}_SH_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{8})T", t)
    h5 = [l["href"] for l in e.get("links", []) if l["href"].startswith("https") and l["href"].endswith(".h5")]
    if m and h5:
        gran.append(dict(track=m.group(1), dir=m.group(2), frame=m.group(3),
                         ref=m.group(4), sec=m.group(5), url=h5[0], titulo=t))
p(f"Granulos GUNW en CMR sobre el AOI: {len(gran)}")

cos = [g for g in gran if g["ref"] <= SISMO <= g["sec"]]
p(f"Pares que CRUZAN el sismo: {len(cos)}")
trabajos = []
for g in sorted(cos, key=lambda g: g["track"]):
    ctrl = [c for c in gran if c["track"] == g["track"] and c["frame"] == g["frame"] and c["sec"] == g["ref"]]
    p(f"  track {g['track']} {g['dir']} frame {g['frame']}: {g['ref']}->{g['sec']} | "
      f"control: {ctrl[0]['ref'] + '->' + ctrl[0]['sec'] if ctrl else 'NO HAY'}")
    trabajos.append((g, "CRUZA"))
    if ctrl: trabajos.append((ctrl[0], "CONTROL"))

# ---------- 2) Lectura por streaming ----------
earthaccess.login(persist=True)

def leer(url):
    for intento in range(1, 5):
        try:
            fobj = earthaccess.open([url])[0]
            with h5py.File(fobj, "r") as f:
                lam = C_LUZ / float(f[f"{ROOT}/centerFrequency"][()])
                x = f[f"{UNW}/xCoordinates"][:]; y = f[f"{UNW}/yCoordinates"][:]
                epsg = int(f[f"{UNW}/projection"][()])
                tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                xa, ya = tr.transform(LON_MIN, LAT_MIN); xb, yb = tr.transform(LON_MAX, LAT_MAX)
                c = np.where((x >= min(xa, xb)) & (x <= max(xa, xb)))[0]
                r = np.where((y >= min(ya, yb)) & (y <= max(ya, yb)))[0]
                if c.size == 0 or r.size == 0:
                    return None
                r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
                d = dict(
                    ph=f[f"{UNW}/HH/unwrappedPhase"][r0:r1, c0:c1].astype(np.float32),
                    io=f[f"{UNW}/HH/ionospherePhaseScreen"][r0:r1, c0:c1].astype(np.float32),
                    co=f[f"{UNW}/HH/coherenceMagnitude"][r0:r1, c0:c1].astype(np.float32),
                    cc=f[f"{UNW}/HH/connectedComponents"][r0:r1, c0:c1].astype(np.int32),
                    mk=f[f"{UNW}/mask"][r0:r1, c0:c1].astype(np.int32),
                    x=x[c0:c1], y=y[r0:r1], epsg=epsg, lam=lam)
            return d
        except Exception as e:
            p(f"    reintento {intento} ({type(e).__name__}: {str(e)[:70]})"); time.sleep(6)
    return None

def mascara_oficial(mk):
    """Igual que MintPy: valido si subswath ref>0 y sec>0 y no es agua (255 = fill)."""
    m8 = np.where(mk == 255, 0, mk & 0xFF)
    agua = (m8 // 100) == 1
    return (mk != 255) & ((m8 // 10) % 10 > 0) & (m8 % 10 > 0) & ~agua

res = {}
for g, tipo in trabajos:
    clave = f"{g['track']}_{g['ref']}_{g['sec']}"
    p(f"\n===== track {g['track']} {g['dir']} {g['ref']}->{g['sec']} [{tipo}] =====")
    d = leer(g["url"])
    if d is None:
        p("  sin datos en el AOI o no se pudo leer"); continue
    ofi = mascara_oficial(d["mk"])
    base = np.isfinite(d["ph"]) & (d["co"] >= 0.2) & (d["cc"] > 0)
    ok = base & ofi & np.isfinite(d["io"])
    p(f"  lambda {d['lam']:.4f} m | validos (coh+cc) {int(base.sum())} | + mascara oficial {int(ok.sum())} | "
      f"fase==0 que SOBREVIVEN a la mascara oficial: {int((ok & (d['ph'] == 0)).sum())}")
    if (ok & (d["ph"] == 0)).sum() > 0:
        p("  -> la mascara oficial NO elimina todos los fase==0: se excluyen igual (criterio conservador)")
        ok &= d["ph"] != 0
    los = np.where(ok, -(d["ph"] - d["io"]) * d["lam"] / (4 * np.pi) * 100, np.nan)  # cm, + = hacia satelite
    los = los - np.nanmedian(los)
    rango = np.nanpercentile(los, 99) - np.nanpercentile(los, 1) if np.isfinite(los).sum() > 100 else np.nan
    p(f"  pixeles usados {int(np.isfinite(los).sum())} | rango p1-p99 {rango:.1f} cm")
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{d['epsg']}", always_xy=True)
    axp, ayp = tr.transform(*AEROPUERTO)
    if d["x"].min() <= axp <= d["x"].max() and d["y"].min() <= ayp <= d["y"].max():
        ix = int(np.abs(d["x"] - axp).argmin()); iy = int(np.abs(d["y"] - ayp).argmin())
        ven = los[iy:iy + 60, max(0, ix - 30):ix + 30]   # ~5 km al sur del aeropuerto (y decrece hacia abajo)
        p(f"  zona sur del aeropuerto: validos {int(np.isfinite(ven).sum())}, "
          f"mediana {np.nanmedian(ven):+.1f} cm (relativo a la mediana de la escena)")
    else:
        p("  el aeropuerto queda fuera de este recorte")
    np.savez_compressed(os.path.join(OUT, f"cos_{clave}.npz"), los_cm=los.astype(np.float32),
                        x=d["x"], y=d["y"], epsg=d["epsg"], lam=d["lam"], co=d["co"], mk=d["mk"])
    res[clave] = dict(g=g, tipo=tipo, los=los, x=d["x"], y=d["y"], epsg=d["epsg"], axp=axp, ayp=ayp)

# ---------- 3) Figura: cada par (colores NASA: rojo = hacia el satelite) ----------
n = len(res)
cols = 2; filas = (n + 1) // 2
fig, ax = plt.subplots(filas, cols, figsize=(14, 4.8 * filas), squeeze=False)
for i, (k, r) in enumerate(res.items()):
    a = ax[i // cols, i % cols]
    ext = [r["x"][0], r["x"][-1], r["y"][-1], r["y"][0]]
    im = a.imshow(r["los"], cmap="RdBu_r", vmin=-30, vmax=30, extent=ext, origin="upper")
    a.plot(r["axp"], r["ayp"], "k*", ms=12)
    a.set_title(f"track {r['g']['track']} {r['g']['dir']} {r['g']['ref']}->{r['g']['sec']} [{r['tipo']}]\n"
                f"LOS cm (+ = hacia el satelite), * aeropuerto", fontsize=9)
    plt.colorbar(im, ax=a)
for j in range(n, filas * cols):
    ax[j // cols, j % cols].axis("off")
plt.tight_layout(); fig.savefig(os.path.join(OUT, "multitrack.png"), dpi=100)
p(f"\nFigura: {os.path.join(OUT, 'multitrack.png')}")

# ---------- 4) Validacion cruzada: dos ascendentes independientes ----------
asc = [k for k, r in res.items() if r["tipo"] == "CRUZA" and r["g"]["dir"] == "A"]
p(f"\nValidacion cruzada entre ascendentes: {asc}")
if len(asc) >= 2:
    A, B = res[asc[0]], res[asc[1]]
    # los tracks pueden caer en zonas UTM distintas (061 = 32620, 162 = 32619): reproyectar
    # cada pixel de A al CRS de B y tomar el vecino mas cercano de B.
    dxB = B["x"][1] - B["x"][0]; dyB = B["y"][1] - B["y"][0]
    XA, YA = np.meshgrid(A["x"], A["y"])
    XB, YB = Transformer.from_crs(f"EPSG:{A['epsg']}", f"EPSG:{B['epsg']}", always_xy=True).transform(XA, YA)
    ixm = np.round((XB - B["x"][0]) / dxB).astype(int); iym = np.round((YB - B["y"][0]) / dyB).astype(int)
    dentro = (ixm >= 0) & (ixm < len(B["x"])) & (iym >= 0) & (iym < len(B["y"]))
    p(f"  EPSG A {A['epsg']} / B {B['epsg']} (reproyectado)")
    Bg = np.full(A["los"].shape, np.nan, np.float32)
    Bg[dentro] = B["los"][iym[dentro], ixm[dentro]]
    comun = np.isfinite(A["los"]) & np.isfinite(Bg)
    p(f"  pixeles en comun: {int(comun.sum())}")
    if comun.sum() > 1000:
        a = A["los"][comun] - np.median(A["los"][comun]); b = Bg[comun] - np.median(Bg[comun])
        r_ = np.corrcoef(a, b)[0, 1]
        pend = np.polyfit(a, b, 1)[0]
        p(f"  correlacion espacial r = {r_:.3f} | pendiente B vs A = {pend:.2f} | "
          f"diferencia mediana absoluta = {np.median(np.abs(a - b)):.1f} cm | rango A {np.ptp(np.percentile(a,[1,99])):.1f} cm")
        p("  Lectura: r alto y pendiente ~1 => los dos pares independientes ven el mismo movimiento.")
        p("  (Diferencias esperables: angulos de incidencia distintos entre tracks, fechas distintas, ruido.)")
        fig2, ax2 = plt.subplots(1, 3, figsize=(17, 5))
        ext = [A["x"][0], A["x"][-1], A["y"][-1], A["y"][0]]
        for a_, img, tit in [(ax2[0], np.where(comun, A["los"], np.nan), asc[0]),
                             (ax2[1], np.where(comun, Bg, np.nan), asc[1])]:
            im = a_.imshow(img, cmap="RdBu_r", vmin=-30, vmax=30, extent=ext); a_.set_title(f"zona comun: {tit}")
            plt.colorbar(im, ax=a_)
        sub = np.random.default_rng(0).choice(a.size, min(20000, a.size), replace=False)
        ax2[2].plot(a[sub], b[sub], ",", alpha=0.3); ax2[2].plot([-40, 40], [-40, 40], "k--", lw=0.8)
        ax2[2].set_xlabel(f"{asc[0]} (cm)"); ax2[2].set_ylabel(f"{asc[1]} (cm)")
        ax2[2].set_title(f"r = {r_:.2f}, pendiente = {pend:.2f}")
        plt.tight_layout(); fig2.savefig(os.path.join(OUT, "cruzada.png"), dpi=100)
        p(f"  Figura: {os.path.join(OUT, 'cruzada.png')}")
else:
    p("  menos de 2 ascendentes con datos: no se puede comparar")
p("\nListo.")
LOG.close()
