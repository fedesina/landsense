"""
sismo_tibet_chequeo_hueco.py -- LandSense
Chequeo de la zona SIN DATO junto al epicentro del sismo del Tibet M7,0 (us6000u43g), orbita 112 A.

Por que: en el par que cruza el sismo (UR, 19 -> 31 oct) hay un hueco sin dato pegado al epicentro con
bordes casi rectos (alarma de artefacto). El 95 % de esos pixeles SI tiene dato en el control
(7 -> 19 oct), asi que no es agua ni una mascara fija: aparece con el sismo. Este script abre las capas
del producto GUNW y clasifica, pixel por pixel, POR QUE src/sismo_generico.py descarto cada pixel del hueco.

Reglas de src/sismo_generico.py para aceptar un pixel (las mismas, sin cambios):
  coherencia >= 0.2 | componente conectado > 0 | mascara oficial valida (sub-swath de ref y sec, no agua)
  | fase desenrollada != 0 | fase e ionosfera finitas.

Interpretacion FIJADA ANTES DE CORRER:
  A) La mayoria del hueco tiene coherencia < 0.2 -> DECORRELACION: el suelo cambio tanto cerca de la
     ruptura que el radar no lo reconoce. Es real (pasa en todo sismo grande). Se muestra "sin dato".
  B) Coherencia >= 0.2 pero componente conectado = 0 -> el DESENROLLADO no pudo seguir la fase (gradiente
     muy fuerte cerca de la falla). Limite del procesamiento, no del terreno; los valores en el BORDE del
     hueco no se citan.
  C) Mascara oficial invalida (sub-swath) -> problema de GEOMETRIA/PROCESAMIENTO del producto: el hueco no
     dice nada del terreno.
  En los tres casos los datos lejos del hueco siguen valiendo (controles planos, C2 r = 0.93-0.96).

Uso:
    python src/sismo_tibet_chequeo_hueco.py
Requiere: login Earthdata (tu propio usuario; el nombre de usuario, no el email) y que src/sismo_generico.py se haya
corrido con --con-ur (usa data/processed/sismos/us6000u43g/pares_los.npz para la grilla).
Prueba con archivos de prueba (sin internet):  python src/sismo_tibet_chequeo_hueco.py --prueba
"""
import os, sys, re, json
print("Cargando librerias (1-2 min en Windows, no cortar)...", flush=True)
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
print("Librerias cargadas.", flush=True)

EVENTO = "us6000u43g"
LON0, LAT0 = 93.7837, 32.3376          # epicentro USGS
R_KM = 25.0                            # radio del recuadro de chequeo
TITULO_CRUZA = "NISAR_L2_UR_GUNW_033_112_A_018_034_2000_SH_20261019T225848_20261019T225923_20261031T225848_20261031T225923_P05023_F_F_J_001"
COL_UR = "C2887502657-ASF"
ROOT = "science/LSAR/GUNW/grids/frequencyA"
UNW = f"{ROOT}/unwrappedInterferogram"
AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "data", "processed", "sismos", EVENTO, "hueco")
os.makedirs(OUT, exist_ok=True)

CAUSAS = ["valido", "sin dato en la grilla (fill)", "mascara: fuera de sub-swath", "mascara: agua",
          "coherencia < 0.2", "componente conectado = 0", "fase = 0 / no finita"]


def clasificar(co, cc, mk, ph):
    """Devuelve un codigo de causa por pixel (primera regla que falla, en el orden de CAUSAS)."""
    causa = np.zeros(co.shape, np.int8)
    m8 = np.where(mk == 255, 0, mk & 0xFF)
    fill = mk == 255
    fuera = (~fill) & (((m8 // 10) % 10 == 0) | (m8 % 10 == 0))
    agua = (~fill) & (~fuera) & ((m8 // 100) == 1)
    resto = ~(fill | fuera | agua)
    baja = resto & ~(co >= 0.2)
    sin_cc = resto & ~baja & ~(cc > 0)
    fase = resto & ~baja & ~sin_cc & ((ph == 0) | ~np.isfinite(ph))
    for k, m in enumerate([fill, fuera, agua, baja, sin_cc, fase], start=1):
        causa[m] = k
    return causa


def leer_recorte(f):
    x = f[f"{UNW}/xCoordinates"][:]; y = f[f"{UNW}/yCoordinates"][:]
    epsg = int(f[f"{UNW}/projection"][()])
    ex, ey = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(LON0, LAT0)
    c = np.where(np.abs(x - ex) <= R_KM * 1000)[0]; r = np.where(np.abs(y - ey) <= R_KM * 1000)[0]
    assert c.size > 10 and r.size > 10, "el recuadro no cae dentro del producto"
    sl = (slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1))
    capa = lambda n, t: f[n][sl].astype(t)
    return dict(x=x[sl[1]], y=y[sl[0]], ex=ex, ey=ey, epsg=epsg,
                co=capa(f"{UNW}/HH/coherenceMagnitude", np.float32),
                cc=capa(f"{UNW}/HH/connectedComponents", np.int32),
                mk=capa(f"{UNW}/mask", np.int32),
                ph=capa(f"{UNW}/HH/unwrappedPhase", np.float32))


def informe(D, nombre="informe.txt"):
    causa = clasificar(D["co"], D["cc"], D["mk"], D["ph"])
    X, Y = np.meshgrid(D["x"], D["y"])
    R = np.hypot(X - D["ex"], Y - D["ey"]) / 1000
    lineas = [f"Chequeo del hueco sin dato, {EVENTO}, orbita 112 A (par UR 19->31 oct)",
              f"Recuadro: +-{R_KM:.0f} km del epicentro | EPSG {D['epsg']}", ""]
    res = {}
    for radio in (5, 10, 25):
        m = R < radio
        tot = int(m.sum()); exc = m & (causa > 0)
        lineas.append(f"A < {radio} km: {tot} px, descartados {int(exc.sum())} ({exc.sum() / max(tot, 1):.0%})")
        res[radio] = {}
        for k in range(1, len(CAUSAS)):
            n = int((m & (causa == k)).sum())
            if n:
                lineas.append(f"    {CAUSAS[k]}: {n} px ({n / max(exc.sum(), 1):.0%} de los descartados)")
            res[radio][CAUSAS[k]] = n
    hueco = (R < 10) & (causa > 0)
    if hueco.sum():
        co_h = D["co"][hueco]
        lineas += ["", f"Coherencia en el hueco (< 10 km): mediana {np.nanmedian(co_h):.2f} | "
                       f"p90 {np.nanpercentile(co_h, 90):.2f}  (afuera, 10-25 km validos: "
                       f"{np.nanmedian(D['co'][(R >= 10) & (causa == 0)]):.2f})"]
    dominante = max(res[10], key=res[10].get) if sum(res[10].values()) else "ninguna"
    letra = {"coherencia < 0.2": "A (decorrelacion: real, se muestra sin dato)",
             "componente conectado = 0": "B (desenrollado: limite del procesamiento; no citar el borde)",
             "mascara: fuera de sub-swath": "C (geometria/procesamiento del producto)"}.get(dominante, "otra: revisar a mano")
    lineas += ["", f"Causa dominante a < 10 km: {dominante}  ->  caso {letra}",
               "(Interpretacion fijada antes de correr: ver encabezado del script.)"]
    open(os.path.join(OUT, nombre), "w", encoding="utf-8").write("\n".join(lineas))
    print("\n".join(lineas))
    return causa, R


def figura(D, causa, nombre="hueco.png"):
    ext = [(D["x"][0] - D["ex"]) / 1e3, (D["x"][-1] - D["ex"]) / 1e3,
           (D["y"][-1] - D["ey"]) / 1e3, (D["y"][0] - D["ey"]) / 1e3]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    im0 = ax[0].imshow(D["co"], cmap="gray", vmin=0, vmax=1, extent=ext)
    ax[0].set_title("Coherencia (0 = el radar no reconoce el suelo)"); fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(np.where(D["cc"] > 0, D["cc"], np.nan), cmap="tab20", extent=ext, interpolation="nearest")
    ax[1].set_title("Componentes conectados (gris = 0, sin desenrollar)"); ax[1].set_facecolor("#888")
    cols = ["#f2efe6", "#000000", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    ax[2].imshow(causa, cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(-0.5, 7.5), 7), extent=ext, interpolation="nearest")
    ax[2].set_title("Por que se descarta cada pixel")
    for k, nom in enumerate(CAUSAS):
        ax[2].plot([], [], "s", color=cols[k], label=nom)
    ax[2].legend(loc="lower left", fontsize=7, framealpha=0.9)
    for a in ax:
        a.plot(0, 0, "*", ms=14, color="yellow", mec="k"); a.set_xlabel("km este"); a.set_ylabel("km norte")
    fig.suptitle(f"{EVENTO}: chequeo del hueco sin dato (orbita 112 A, par UR 19->31 oct)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, nombre), dpi=110); plt.close(fig)
    print("Figura:", os.path.join(OUT, nombre))


def prueba():
    """Archivos de prueba: un hueco de baja coherencia (A) y uno de cc = 0 (B). Se borran al terminar."""
    n = 101; x = np.linspace(-25e3, 25e3, n); y = np.linspace(25e3, -25e3, n)
    co = np.full((n, n), 0.6, np.float32); cc = np.ones((n, n), np.int32)
    mk = np.full((n, n), 11, np.int32); ph = np.ones((n, n), np.float32)
    co[45:55, 45:55] = 0.05          # hueco A: decorrelacion
    cc[40:44, 40:60] = 0             # franja B: sin desenrollar
    D = dict(x=x, y=y, ex=0.0, ey=0.0, epsg=32646, co=co, cc=cc, mk=mk, ph=ph)
    causa, R = informe(D, "prueba_informe.txt")
    assert (causa[47, 47] == CAUSAS.index("coherencia < 0.2")), "hueco A mal clasificado"
    assert (causa[41, 50] == CAUSAS.index("componente conectado = 0")), "franja B mal clasificada"
    assert (causa[10, 10] == 0), "pixel valido mal clasificado"
    os.remove(os.path.join(OUT, "prueba_informe.txt"))
    print("[prueba] OK: decorrelacion, desenrollado y pixeles validos se clasifican bien.")


if __name__ == "__main__":
    if "--prueba" in sys.argv:
        prueba(); sys.exit(0)
    import h5py, earthaccess
    u = (f"https://cmr.earthdata.nasa.gov/search/granules.json?collection_concept_id={COL_UR}"
         f"&readable_granule_name={TITULO_CRUZA}&page_size=5")
    ent = requests.get(u, timeout=60).json()["feed"]["entry"]
    assert ent, "no se encontro el granulo UR en CMR"
    h5 = [l["href"] for l in ent[0]["links"] if l["href"].startswith("https") and l["href"].endswith(".h5")
          and "QA_STATS" not in l["href"]]
    assert h5, "el granulo no trae .h5"
    earthaccess.login(persist=True)
    print("Leyendo capas del producto por streaming (puede tardar unos minutos)...", flush=True)
    with h5py.File(earthaccess.open([h5[0]])[0], "r") as f:
        D = leer_recorte(f)
    causa, R = informe(D)
    figura(D, causa)
    json.dump({"evento": EVENTO, "granulo": TITULO_CRUZA}, open(os.path.join(OUT, "fuente.json"), "w"))
