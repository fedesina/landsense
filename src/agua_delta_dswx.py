"""
agua_delta_dswx.py -- LandSense
Verifica con los mapas de agua OFICIALES de NASA (OPERA DSWx) si el cambio que vio NISAR en el Delta
del Parana (eco que se apago en agosto, src/agua_gcov_exploratorio.py y src/agua_delta_export_visor.py) es AGUA.

Productos (PO.DAAC, 30 m, mosaicos MGRS 21HTC y 21HUC). OJO: vienen en EPSG:32621 (UTM 21 "norte", y negativas)
aunque la zona es del sur; NISAR esta en EPSG:32721. Se corrige y en +10.000.000 m (misma zona y grilla).
Un assert de proyeccion frena el calculo si las grillas no coinciden, antes de dar un resultado falso.
  - OPERA_L3_DSWX-HLS_V1 (optico Landsat/Sentinel-2): independiente del radar; depende de las nubes.
  - OPERA_L3_DSWX-S1_V1  (radar Sentinel-1, banda C): ve a traves de nubes; otro satelite/frecuencia.
Capa usada: B01_WTR. Clases leidas del archivo y reportadas (protocolo B); se usan:
  agua = 1 (agua abierta) y 2 (agua parcial, solo HLS); no agua = 0; el resto = sin dato (nube, relleno, etc.).

Diseno (fijado ANTES de ver el resultado):
  - Grupo MARCADO: celdas NISAR de 100 m donde el HV se apago >= 3 dB el 25-ago (pico) respecto de la
    media 26-jun/8-jul/20-jul (mismo criterio del visor).
  - Grupo CONTROL: celdas donde |cambio| < 1 dB ese dia.
  - Para cada fecha DSWx: % de agua en cada grupo (sobre pixeles DSWx validos).
  - Prediccion si es agua: MARCADO sube en agosto y baja en septiembre; CONTROL casi constante;
    antes del evento (junio-julio) ambos parecidos. Si MARCADO no sube -> la hipotesis de agua NO se confirma.
  - Fechas con < 50% de pixeles validos en la zona (nubes) se reportan pero no se usan para concluir.
Probado: busqueda CMR real (45 fechas HLS, 18 S1) y agregacion 30 m -> 100 m con raster sintetico.

Requiere: src/agua_gcov_exploratorio.py corrido (delta_parana.npz). Login Earthdata (tu propio usuario; el nombre de usuario, no el email).
Uso:
    python src/agua_delta_dswx.py
Salidas: ../data/processed/gcov_exploratorio/dswx_vs_nisar.txt, dswx_vs_nisar.png, dswx_serie.json
"""
print("Iniciando chequeo DSWx... (cargar librerias puede tardar 1-2 min)", flush=True)
import os, re, json, time
from collections import defaultdict
import numpy as np
import requests
import earthaccess
import rasterio
from rasterio.io import MemoryFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("Librerias cargadas.", flush=True)

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(AQUI, "..", "data", "processed", "gcov_exploratorio")
BBOX = "-59.3,-34.1,-58.8,-33.6"
PRODUCTOS = {"HLS (optico)": "C2617126679-POCLOUD", "S1 (radar C)": "C2949811996-POCLOUD"}
MIN_VALIDO = 0.5
LOG = open(os.path.join(DIR, "dswx_vs_nisar.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---------- grupos NISAR ----------
z = np.load(os.path.join(DIR, "delta_parana.npz"))
HV = z["HV"]; fechas_n = [str(f) for f in z["fechas"]]
x0, y0, dx, dy, epsg = float(z["x0"]), float(z["y0"]), float(z["dx"]), float(z["dy"]), int(z["epsg"])
ny, nx = HV.shape[1:]
paso_orig = abs(dx) / round(abs(dx) / 20.0)
# borde del bloque (no el centro): x0/y0 son centros del primer pixel de 20 m
bx0 = x0 - np.sign(dx) * paso_orig / 2; by0 = y0 - np.sign(dy) * paso_orig / 2
ref = np.nanmean(HV[:3], 0)
d_pico = 10 * np.log10(HV[fechas_n.index("20260825")] / ref)
MARCADO = np.isfinite(d_pico) & (d_pico <= -3)
CONTROL = np.isfinite(d_pico) & (np.abs(d_pico) < 1)
p(f"NISAR (EPSG:{epsg}): celdas MARCADAS {MARCADO.sum()} | CONTROL {CONTROL.sum()} | grilla {ny}x{nx} de 100 m")
assert epsg == 32721, "se esperaba UTM 21S como los mosaicos DSWx 21HTC/21HUC"

# ---------- granulos DSWx ----------
def granulos(cid):
    out, page = [], 1
    while True:
        r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", timeout=60, params=dict(
            collection_concept_id=cid, bounding_box=BBOX, temporal="2026-06-15T00:00:00Z,2026-09-25T00:00:00Z",
            page_size=200, page_num=page)).json()["feed"]["entry"]
        out += r
        if len(r) < 200:
            return out
        page += 1

def por_fecha(E):
    """(fecha) -> {mosaico: url B01_WTR}, quedandose con el procesamiento mas reciente."""
    mejor = {}
    for e in E:
        m = re.search(r"_T(\w{5})_(\d{8})T\d{6}Z_(\d{8}T\d{6})Z", e["title"])
        url = [l["href"] for l in e["links"] if l["href"].endswith("_B01_WTR.tif") and l["href"].startswith("https")]
        if not (m and url):
            continue
        k = (m.group(2), m.group(1))
        if k not in mejor or m.group(3) > mejor[k][0]:
            mejor[k] = (m.group(3), url[0])
    res = defaultdict(dict)
    for (f, t), (_, u) in mejor.items():
        res[f][t] = u
    return dict(sorted(res.items()))

earthaccess.login(persist=True)
valores_vistos = defaultdict(set)

# Los DSWx vienen en UTM 21 "norte" (EPSG:32621) aun en el hemisferio sur (y negativas); NISAR en 21 sur
# (EPSG:32721). Misma zona: x igual, y_sur = y_norte + 10.000.000 m. Otra proyeccion -> error, sin reintentar.
DESPLAZ_Y = {32721: 0.0, 32621: 10_000_000.0}
epsg_vistos = defaultdict(set)

def leer_wtr(url, nombre):
    for intento in range(1, 5):
        try:
            data = earthaccess.open([url])[0].read()
        except Exception as ex:
            p(f"    reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})"); time.sleep(5); continue
        with MemoryFile(data) as mf, mf.open() as ds:
            e = ds.crs.to_epsg()
            if e not in DESPLAZ_Y:
                raise ValueError(f"proyeccion inesperada {ds.crs} en {url}")
            a = ds.read(1); tr = ds.transform
        valores_vistos[nombre].update(np.unique(a).tolist()); epsg_vistos[nombre].add(e)
        return a, tr, DESPLAZ_Y[e]
    return None, None, None

def a_celdas(a, tr, agua_vals, desplaz_y=0.0):
    """Suma pixeles de agua y validos DSWx (30 m) por celda NISAR de 100 m."""
    agua = np.zeros(ny * nx); val = np.zeros(ny * nx)
    filas, cols = np.indices(a.shape)
    xs = tr.c + (cols + 0.5) * tr.a; ys = tr.f + (filas + 0.5) * tr.e + desplaz_y
    ci = np.floor((xs - bx0) / dx).astype(int); ri = np.floor((ys - by0) / dy).astype(int)
    ok = (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny)
    es_agua = np.isin(a, agua_vals); es_val = es_agua | (a == 0)
    idx = (ri * nx + ci)[ok]
    agua += np.bincount(idx, weights=es_agua[ok], minlength=ny * nx)
    val += np.bincount(idx, weights=es_val[ok], minlength=ny * nx)
    return agua, val

serie = {}
for nombre, cid in PRODUCTOS.items():
    agua_vals = [1, 2] if "HLS" in nombre else [1]
    F = por_fecha(granulos(cid))
    p(f"\n===== DSWx {nombre}: {len(F)} fechas")
    p("fecha      validos  %agua MARCADO  %agua CONTROL  %agua toda la zona")
    serie[nombre] = []
    for f, tiles in F.items():
        agua = np.zeros(ny * nx); val = np.zeros(ny * nx)
        for t, u in tiles.items():
            a, tr, dy_ = leer_wtr(u, nombre)
            if a is None:
                continue
            ag, va = a_celdas(a, tr, agua_vals, dy_); agua += ag; val += va
        agua = agua.reshape(ny, nx); val = val.reshape(ny, nx)
        cubierta = val > 0
        frac_valida = float(cubierta[np.isfinite(ref)].mean())
        pct = lambda M: float(100 * agua[M & cubierta].sum() / max(val[M & cubierta].sum(), 1))
        fila = dict(fecha=f, validos=frac_valida, marcado=pct(MARCADO), control=pct(CONTROL), zona=pct(np.isfinite(ref)))
        serie[nombre].append(fila)
        aviso = "" if frac_valida >= MIN_VALIDO else "  (pocas celdas validas: nubes -> no se usa)"
        p(f"{f}  {frac_valida:5.0%}   {fila['marcado']:8.1f}      {fila['control']:8.1f}       {fila['zona']:8.1f}{aviso}")
    p(f"  valores de clase encontrados en B01_WTR ({nombre}): {sorted(valores_vistos[nombre])} | EPSG: {sorted(epsg_vistos[nombre])}")

json.dump(serie, open(os.path.join(DIR, "dswx_serie.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- figura ----------
fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for j, (nombre, filas) in enumerate(serie.items()):
    buenas = [r for r in filas if r["validos"] >= MIN_VALIDO]
    xs = [np.datetime64(f"{r['fecha'][:4]}-{r['fecha'][4:6]}-{r['fecha'][6:]}") for r in buenas]
    ax[j].plot(xs, [r["marcado"] for r in buenas], "o-", color="#2166ac", label="zonas que NISAR marco (se apago el eco)")
    ax[j].plot(xs, [r["control"] for r in buenas], "s-", color="gray", label="zonas sin cambio (control)")
    for fn in fechas_n:
        ax[j].axvline(np.datetime64(f"{fn[:4]}-{fn[4:6]}-{fn[6:]}"), color="k", alpha=0.12)
    ax[j].axvspan(np.datetime64("2026-07-27"), np.datetime64("2026-08-10"), color="orange", alpha=0.15, label="hueco NISAR")
    ax[j].set_title(f"OPERA DSWx {nombre}: % de agua"); ax[j].legend(fontsize=8); ax[j].tick_params(axis="x", rotation=45)
ax[0].set_ylabel("% de pixeles clasificados como agua")
plt.tight_layout(); fig.savefig(os.path.join(DIR, "dswx_vs_nisar.png"), dpi=90)
p(f"\nFigura: {os.path.join(DIR, 'dswx_vs_nisar.png')}")
LOG.close()
