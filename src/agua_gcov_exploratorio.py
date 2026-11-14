"""
agua_gcov_exploratorio.py -- LandSense
PRUEBA EXPLORATORIA (1 dia) de la otra mitad de NISAR: el eco GCOV (HH + HV) con las
RECETAS OFICIALES de la mision, sobre dos zonas argentinas.

Recetas (no inventadas):
  HUMEDALES -- notebook oficial NISAR_ATBD_wetlands.ipynb (Chapman & Siqueira; ASFOpenSARlab,
    NISAR Early Adopters Workshop Oct-2024), seccion 3.2 "Classify Time Series", potencia lineal gamma-0:
      agua abierta          : 1e-6 < HH*HV < 5e-5
      vegetacion inundada A : 0.5 < HH < 10   y  6 < HH/HV < 500   (clase 16, HH muy brillante)
      vegetacion inundada B : 0.1 < HH < 0.3  y  8 < HH/HV < 500   (clase 12)
    El notebook aclara: "Thresholds to be calibrated based on other data sets" -> son valores por
    defecto de la mision, NO calibrados para Argentina.
    Calibracion relativa entre fechas como el notebook (ical=1): HH y HV de cada fecha divididos por
    (HV medio de esa fecha / HV medio de toda la serie).
  CULTIVOS -- algoritmo de area de cultivo NISAR: coeficiente de variacion (CV = desvio/media) de
    la serie temporal de HV en potencia lineal; cultivo activo = CV alto; bosque/estable tipicamente
    CV < 0.2 (Remote Sensing 16(11):1975, 2024; Huang et al. 2021 RSE). A 100 m rinde mejor el HV.
    Umbral de clasificacion: NO se fija aca (la mision lo optimiza con un mapa de referencia).

Control cruzado incorporado:
  - la receta de humedales se aplica tambien a la PAMPA (campo agricola): deberia dar poca
    vegetacion inundada; si da mucha, los umbrales no sirven tal cual aca.
  - la receta de cultivos se aplica tambien al DELTA, para comparar comportamientos.

Promedio espacial a ~100 m (bloques en potencia lineal) antes de todo: baja el speckle y el volumen.
Mascara: pixeles con HH > 0 y HV > 0 finitos (como el notebook). La capa 'mask' del GCOV NO se usa
(su semantica no esta verificada por nosotros).
Probado con datos sinteticos (clases y CV correctos) y busqueda CMR real (6-7 fechas por zona).

Uso:
    python src/agua_gcov_exploratorio.py
Salidas: ../data/processed/gcov_exploratorio/<zona>.png, informe.txt, <zona>.npz
"""
print("Iniciando prueba GCOV... (cargar librerias puede tardar 1-2 min)", flush=True)
import os, re, time
from collections import defaultdict
import numpy as np
import requests
import h5py
import earthaccess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
print("Librerias cargadas.", flush=True)

COLECCION = "C2854338529-ASF"          # NISAR_L2_GCOV_PROVISIONAL_V1
G = "science/LSAR/GCOV/grids/frequencyA"
ZONAS = {
    "pampa_pergamino": dict(bbox=(-60.8, -34.0, -60.2, -33.6), tipo="cultivos"),
    "delta_parana":    dict(bbox=(-59.3, -34.1, -58.8, -33.6), tipo="humedal"),
}
OBJ_M = 100.0                          # tamanio de promedio espacial
T = dict(ow_min=1e-6, ow_max=5e-5, a_hh_min=0.5, a_hh_max=10.0, a_r_min=6, a_r_max=500,
         b_hh_min=0.1, b_hh_max=0.3, b_r_min=8, b_r_max=500)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "gcov_exploratorio")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "informe.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---------- funciones de las recetas (probadas con datos sinteticos) ----------
def bloques(a, k):
    """Promedio en bloques k x k (potencia lineal), ignorando NaN."""
    ny, nx = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:ny, :nx].reshape(ny // k, k, nx // k, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))

def calibrar(HH, HV):
    """Calibracion relativa entre fechas como el notebook oficial (ical=1)."""
    m_all = np.nanmean(HV)
    f = np.array([np.nanmean(HV[i]) / m_all for i in range(len(HV))])
    return HH / f[:, None, None], HV / f[:, None, None], f

def clasificar_humedal(hh, hv):
    """0 sin dato, 8 sin agua, 3 agua abierta, 16 veg. inundada (HH brillante), 12 veg. inundada (HH/HV alto)."""
    c = np.zeros(hh.shape, np.int8)
    ok = np.isfinite(hh) & np.isfinite(hv) & (hh > 0) & (hv > 0)
    r = np.where(ok, hh / np.where(hv > 0, hv, np.nan), np.nan); pr = hh * hv
    c[ok] = 8
    c[ok & (pr > T["ow_min"]) & (pr < T["ow_max"])] = 3
    c[ok & (c != 3) & (hh > T["a_hh_min"]) & (hh < T["a_hh_max"]) & (r > T["a_r_min"]) & (r < T["a_r_max"])] = 16
    c[ok & (c != 3) & (hh > T["b_hh_min"]) & (hh < T["b_hh_max"]) & (r > T["b_r_min"]) & (r < T["b_r_max"])] = 12
    return c

def cv_serie(HV):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanstd(HV, 0) / np.nanmean(HV, 0)

# ---------- busqueda de fechas (CMR directo) ----------
def buscar(bbox):
    w, s, e, n = bbox
    url = (f"https://cmr.earthdata.nasa.gov/search/granules.json?collection_concept_id={COLECCION}"
           f"&bounding_box={w},{s},{e},{n}&temporal=2026-06-17T00:00:00Z,2026-12-31T00:00:00Z&page_size=500")
    grupos = defaultdict(list)
    for ent in requests.get(url, timeout=60).json()["feed"]["entry"]:
        m = re.search(r"GCOV_\d{3}_(\d{3})_([AD])_(\d{3})_\d{4}_(\w{4})_[A-Z]_(\d{8})T", ent["title"])
        h5 = [l["href"] for l in ent.get("links", []) if l["href"].startswith("https") and l["href"].endswith(".h5")]
        if m and h5 and m.group(4) == "DHDH":
            grupos[(m.group(1), m.group(2), m.group(3))].append(dict(fecha=m.group(5), url=h5[0]))
    return grupos

def leer(u, bbox):
    w, s, e, n = bbox
    for intento in range(1, 5):
        try:
            with h5py.File(earthaccess.open([u])[0], "r") as f:
                x = f[f"{G}/xCoordinates"][:]; y = f[f"{G}/yCoordinates"][:]
                epsg = int(f[f"{G}/projection"][()])
                tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                xa, ya = tr.transform(w, s); xb, yb = tr.transform(e, n)
                c = np.where((x >= min(xa, xb)) & (x <= max(xa, xb)))[0]
                r = np.where((y >= min(ya, yb)) & (y <= max(ya, yb)))[0]
                if c.size < 50 or r.size < 50:
                    return None
                r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
                hh = f[f"{G}/HHHH"][r0:r1, c0:c1].astype(np.float32)
                hv = f[f"{G}/HVHV"][r0:r1, c0:c1].astype(np.float32)
            paso = abs(float(x[1] - x[0])); k = max(1, int(round(OBJ_M / paso)))
            bad = ~(np.isfinite(hh) & np.isfinite(hv) & (hh > 0) & (hv > 0))
            hh[bad] = np.nan; hv[bad] = np.nan
            return dict(hh=bloques(hh, k), hv=bloques(hv, k), paso=paso, k=k, epsg=epsg,
                        x0=float(x[c0]), y0=float(y[r0]), dx=float(x[1] - x[0]) * k, dy=float(y[1] - y[0]) * k)
        except Exception as ex:
            p(f"    reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})"); time.sleep(6)
    return None

# ---------- main ----------
if __name__ == "__main__":
    earthaccess.login(persist=True)
    for zona, cfg in ZONAS.items():
        bbox = cfg["bbox"]
        p(f"\n==================== {zona} ({cfg['tipo']}) bbox {bbox}")
        gr = buscar(bbox)
        if not gr:
            p("  sin granulos DHDH"); continue
        clave = max(gr, key=lambda k: len({g['fecha'] for g in gr[k]}))
        gs = sorted({g["fecha"]: g for g in gr[clave]}.values(), key=lambda g: g["fecha"])
        p(f"  geometria elegida: track {clave[0]} {clave[1]} frame {clave[2]} | {len(gs)} fechas: {', '.join(g['fecha'] for g in gs)}")
        pilas, fechas = [], []
        for g in gs:
            d = leer(g["url"], bbox)
            if d is None:
                p(f"  {g['fecha']}: sin datos en la zona"); continue
            if pilas and d["hh"].shape != pilas[0]["hh"].shape:
                p(f"  {g['fecha']}: grilla distinta, se omite"); continue
            pilas.append(d); fechas.append(g["fecha"])
            p(f"  {g['fecha']}: pixel original {d['paso']:.0f} m -> promedio {d['k']}x{d['k']} (~{d['paso']*d['k']:.0f} m) | "
              f"HH mediana {10*np.log10(np.nanmedian(d['hh'])):.1f} dB, HV mediana {10*np.log10(np.nanmedian(d['hv'])):.1f} dB")
        if len(pilas) < 4:
            p("  menos de 4 fechas utilizables: no se analiza"); continue
        HH = np.stack([d["hh"] for d in pilas]); HV = np.stack([d["hv"] for d in pilas])
        HHc, HVc, fac = calibrar(HH, HV)
        p(f"  factores de calibracion relativa por fecha (HV): {np.round(fac, 3).tolist()}")

        # HUMEDALES por fecha
        C = np.stack([clasificar_humedal(HHc[i], HVc[i]) for i in range(len(fechas))])
        p("  HUMEDALES (receta oficial, umbrales por defecto) - % de pixeles validos por fecha:")
        p("    fecha      agua abierta   veg.inundada(16)   veg.inundada(12)   sin agua")
        for i, fe in enumerate(fechas):
            v = C[i] > 0; t = max(v.sum(), 1)
            p(f"    {fe}   {100*(C[i]==3).sum()/t:8.1f}      {100*(C[i]==16).sum()/t:10.1f}        {100*(C[i]==12).sum()/t:10.1f}     {100*(C[i]==8).sum()/t:6.1f}")
        inund = ((C == 16) | (C == 12)).sum(0) / np.maximum((C > 0).sum(0), 1)
        agua = (C == 3).sum(0) / np.maximum((C > 0).sum(0), 1)

        # CULTIVOS
        cv = cv_serie(HVc)
        v = np.isfinite(cv)
        p(f"  CULTIVOS (CV de HV): mediana {np.nanmedian(cv):.2f} | p10 {np.nanpercentile(cv,10):.2f} | p90 {np.nanpercentile(cv,90):.2f} | "
          f"% pixeles con CV > 0.2: {100*np.mean(cv[v] > 0.2):.0f}%")
        alto = cv >= np.nanpercentile(cv, 90); bajo = cv <= np.nanpercentile(cv, 10)
        serie_alto = [10*np.log10(np.nanmedian(HVc[i][alto])) for i in range(len(fechas))]
        serie_bajo = [10*np.log10(np.nanmedian(HVc[i][bajo])) for i in range(len(fechas))]
        p("  HV mediano (dB) por fecha - 10% mas variable vs 10% menos variable:")
        for i, fe in enumerate(fechas):
            p(f"    {fe}: variables {serie_alto[i]:6.1f} | estables {serie_bajo[i]:6.1f}")

        np.savez_compressed(os.path.join(OUT, f"{zona}.npz"), HH=HHc.astype(np.float32), HV=HVc.astype(np.float32),
                            clases=C, cv=cv.astype(np.float32), fechas=np.array(fechas),
                            x0=pilas[0]["x0"], y0=pilas[0]["y0"], dx=pilas[0]["dx"], dy=pilas[0]["dy"], epsg=pilas[0]["epsg"])

        # figura
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        ax[0, 0].imshow(10*np.log10(np.nanmean(HVc, 0)), cmap="gray", vmin=-30, vmax=-10); ax[0, 0].set_title("HV medio (dB)")
        ax[0, 1].imshow(10*np.log10(np.nanmean(HHc, 0) / np.nanmean(HVc, 0)), cmap="viridis", vmin=0, vmax=12); ax[0, 1].set_title("HH/HV medio (dB) - alto = posible rebote doble")
        im = ax[0, 2].imshow(cv, cmap="magma", vmin=0, vmax=1); ax[0, 2].set_title("CULTIVOS: CV de HV (alto = cambia en el tiempo)"); plt.colorbar(im, ax=ax[0, 2])
        cmap = matplotlib.colors.ListedColormap(["white", "lightgray", "royalblue", "seagreen", "darkorange"])
        cod = lambda c: np.select([c == 0, c == 8, c == 3, c == 16, c == 12], [0, 1, 2, 3, 4])
        ax[1, 0].imshow(cod(C[0]), cmap=cmap, vmin=0, vmax=4, interpolation="nearest"); ax[1, 0].set_title(f"HUMEDAL {fechas[0]}\ngris sin agua, azul agua, verde/naranja veg. inundada")
        ax[1, 1].imshow(cod(C[-1]), cmap=cmap, vmin=0, vmax=4, interpolation="nearest"); ax[1, 1].set_title(f"HUMEDAL {fechas[-1]}")
        ax[1, 2].plot(range(len(fechas)), serie_alto, "o-", label="10% mas variable"); ax[1, 2].plot(range(len(fechas)), serie_bajo, "s-", label="10% mas estable")
        ax[1, 2].set_xticks(range(len(fechas))); ax[1, 2].set_xticklabels([f[4:] for f in fechas], rotation=45); ax[1, 2].set_ylabel("HV (dB)"); ax[1, 2].legend(); ax[1, 2].set_title("Serie temporal HV")
        fig.suptitle(f"{zona} - track {clave[0]} {clave[1]} - GCOV PROVISIONAL, ~100 m", fontsize=12)
        plt.tight_layout(); fig.savefig(os.path.join(OUT, f"{zona}.png"), dpi=85); plt.close(fig)
        p(f"  figura: {os.path.join(OUT, zona + '.png')}")
    LOG.close()
