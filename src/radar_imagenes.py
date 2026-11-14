"""
radar_imagenes.py -- LandSense
SOLO IMAGENES (no afirma nada, no tiene criterios): muestra lo que vio el radar NISAR (GCOV, banda L)
fecha por fecha sobre una zona, para mirar a ojo como cambia el agua.

Composicion en colores falsos, MISMA escala fija en todas las fechas (asi son comparables):
  rojo  = HH        (-25 .. -5 dB)   -> superficies "duras", rebote doble, ciudad
  verde = HV        (-30 .. -12 dB)  -> volumen: vegetacion, monte
  azul  = HH/HV     (  0 .. 15 dB) x brillo -> suelo liso / poco volumen (se apaga si no hay eco)
  NEGRO = agua lisa (espejo: casi no vuelve eco). Gris oscuro/violeta = campo. Verde = monte.
Resolucion: pixel nativo GCOV (20 m) con un suavizado 3x3 (baja el moteado, ~60 m efectivo).
En la pagina debe decir: "imagen del radar, sin validar".

Uso:
    python src/radar_imagenes.py                 (todas las zonas)
    python src/radar_imagenes.py apipe iguazu    (algunas)
    python src/radar_imagenes.py --prueba     (sintetico, sin internet)
Salidas: ../data/processed/imagenes_radar/<zona>/<fecha>.png  y  <zona>_hoja.png  (todas las fechas juntas)
Login Earthdata: tu propio usuario (el nombre de usuario, NO el email).
"""
import os, re, sys, time
from collections import defaultdict
import numpy as np

GCOV = "C2854338529-ASF"
G = "science/LSAR/GCOV/grids/frequencyA"
ZONAS = {
    "apipe":      (-56.95, -27.60, -56.55, -27.40),   # isla Apipe - Ituzaingo, aguas abajo de Yacyreta
    "itati":      (-58.40, -27.35, -58.15, -27.20),   # Itati - Paso de la Patria
    "concordia":  (-58.20, -31.60, -57.80, -31.10),   # rio Uruguay (prueba src/agua_crecida_uruguay.py)
    "libres":     (-57.25, -29.85, -56.95, -29.55),   # Paso de los Libres (prueba src/agua_crecida_uruguay.py)
    "iguazu":     (-54.47, -25.72, -54.40, -25.66),   # Cataratas (canion angosto: esperar sombras radar)
    "gran_resistencia": (-59.05, -27.60, -58.55, -27.00),  # crecida Parana/Paraguay oct-2026 (Barranqueras, Las Palmas, Corrientes)
}
ESCALA = dict(R=(-20.0, -3.0), G=(-25.0, -10.0), B=(0.0, 15.0))   # agua mas negra (HV del agua ~ piso de ruido)
MAX_FECHAS_HOJA = 16
FECHAS = []   # opcional: --fechas 20260901,20261019 (solo esas)

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "data", "processed", "imagenes_radar")

def suavizar(a, k=3):
    """Media movil k x k ignorando NaN (solo numpy)."""
    v = np.isfinite(a); x = np.where(v, a, 0.0)
    r = k // 2
    xp = np.pad(x, r, mode="constant"); vp = np.pad(v.astype(float), r, mode="constant")
    S = sum(xp[i:i + a.shape[0], j:j + a.shape[1]] for i in range(k) for j in range(k))
    N = sum(vp[i:i + a.shape[0], j:j + a.shape[1]] for i in range(k) for j in range(k))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = S / N
    out[~v] = np.nan
    return out

def rgb(hh, hv):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 10 * np.log10(hh); g = 10 * np.log10(hv); b = r - g
    def esc(x, lo, hi): return np.clip((x - lo) / (hi - lo), 0, 1)
    R, Gc = esc(r, *ESCALA["R"]), esc(g, *ESCALA["G"])
    B = esc(b, *ESCALA["B"]) * np.maximum(R, Gc)      # azul se apaga donde no vuelve eco (agua = negro)
    img = np.dstack([R, Gc, B])
    img[~(np.isfinite(r) & np.isfinite(g))] = 1.0      # sin dato = blanco
    return img

def prueba():
    rng = np.random.default_rng(0); n = 120
    hh = 10 ** (rng.normal(-12, 1, (n, n)) / 10); hv = 10 ** (rng.normal(-19, 1, (n, n)) / 10)   # campo
    hh[:, :30] = 10 ** (-28 / 10); hv[:, :30] = 10 ** (-33 / 10)                                   # agua
    hh[:, 90:] = 10 ** (-9 / 10);  hv[:, 90:] = 10 ** (-13 / 10)                                    # monte
    hh[0:5, 0:5] = np.nan
    im = rgb(suavizar(hh), suavizar(hv))
    agua, campo, monte = im[60, 10], im[60, 60], im[60, 110]
    print("[prueba] agua", agua.round(2), "| campo", campo.round(2), "| monte", monte.round(2))
    assert agua.max() < 0.25, "el agua debe salir casi negra"
    assert monte[1] > campo[1] and monte[1] > 0.5, "el monte debe ser mas verde que el campo"
    assert (im[0, 0] == 1).all(), "sin dato = blanco"
    print("[prueba] OK: agua negra, monte verde, sin dato blanco.")

def buscar(bbox):
    import requests
    w, s, e, n = bbox
    r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", timeout=60, params=dict(
        collection_concept_id=GCOV, bounding_box=f"{w},{s},{e},{n}",
        temporal="2026-06-17T00:00:00Z,2026-12-31T00:00:00Z", page_size=500)).json()["feed"]["entry"]
    g = defaultdict(dict)
    for ent in r:
        m = re.search(r"GCOV_\d{3}_(\d{3})_([AD])_(\d{3})_\d{4}_(\w{4})_[A-Z]_(\d{8})T.*_(\d{3})$", ent["title"])
        h5 = [l["href"] for l in ent.get("links", []) if l["href"].startswith("https") and l["href"].endswith(".h5")]
        if m and h5 and m.group(4) == "DHDH":
            k = (m.group(1), m.group(2), m.group(3)); f = m.group(5)
            if f not in g[k] or m.group(6) > g[k][f][0]:
                g[k][f] = (m.group(6), h5[0])
    return {k: {f: u for f, (v, u) in d.items()} for k, d in g.items()}

def leer(u, bbox):
    import h5py, earthaccess
    from pyproj import Transformer
    w, s, e, n = bbox
    for intento in range(1, 5):
        try:
            with h5py.File(earthaccess.open([u])[0], "r") as f:
                x = f[f"{G}/xCoordinates"][:]; y = f[f"{G}/yCoordinates"][:]
                epsg = int(f[f"{G}/projection"][()])
                tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                xs, ys = tr.transform([w, e, w, e], [s, s, n, n])
                c = np.where((x >= min(xs)) & (x <= max(xs)))[0]; r = np.where((y >= min(ys)) & (y <= max(ys)))[0]
                if c.size < 20 or r.size < 20: return None
                hh = f[f"{G}/HHHH"][r.min():r.max() + 1, c.min():c.max() + 1].astype(np.float32)
                hv = f[f"{G}/HVHV"][r.min():r.max() + 1, c.min():c.max() + 1].astype(np.float32)
            bad = ~(np.isfinite(hh) & np.isfinite(hv) & (hh > 0) & (hv > 0)); hh[bad] = np.nan; hv[bad] = np.nan
            return hh, hv, abs(float(x[1] - x[0]))
        except Exception as ex:
            print(f"    reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})", flush=True); time.sleep(6)
    return None

def zona(nombre, bbox):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print(f"\n===== {nombre} {bbox}", flush=True)
    orb = buscar(bbox)
    if not orb: print("  sin GCOV DHDH"); return
    for k, d in sorted(orb.items()): print(f"  orbita {k[0]} {k[1]} frame {k[2]}: {len(d)} fechas")
    k = max(orb, key=lambda k: (len(orb[k]), max(orb[k])))
    fechas = sorted(orb[k])
    if FECHAS:
        fechas = [f for f in fechas if f in FECHAS]
        print(f"  solo las fechas pedidas: {fechas}", flush=True)
    print(f"  ELEGIDA {k[0]} {k[1]} frame {k[2]} (mas fechas): {fechas}", flush=True)
    od = os.path.join(OUT, nombre); os.makedirs(od, exist_ok=True)
    imgs = []
    for i, f in enumerate(fechas, 1):
        t0 = time.time(); print(f"  {f}: bajando ({i}/{len(fechas)})... puede tardar varios minutos, no cortar", flush=True)
        r = leer(orb[k][f], bbox)
        print(f"     listo en {time.time() - t0:.0f} s", flush=True)
        if r is None: print(f"  {f}: sin datos en la zona"); continue
        hh, hv, paso = r
        im = rgb(suavizar(hh), suavizar(hv))
        val = np.isfinite(hh).mean()
        plt.imsave(os.path.join(od, f"{f}.png"), im)
        imgs.append((f, im)); print(f"  {f}: {im.shape[1]}x{im.shape[0]} px de {paso:.0f} m | con dato {val:.0%}", flush=True)
    if not imgs: return
    sel = imgs if len(imgs) <= MAX_FECHAS_HOJA else [imgs[i] for i in np.linspace(0, len(imgs) - 1, MAX_FECHAS_HOJA).astype(int)]
    nc = min(4, len(sel)); nr = int(np.ceil(len(sel) / nc))
    asp = sel[0][1].shape[0] / sel[0][1].shape[1]
    fig, ax = plt.subplots(nr, nc, figsize=(4.5 * nc, 4.5 * nr * asp + 1.2), squeeze=False)
    for a in ax.ravel(): a.axis("off")
    for a, (f, im) in zip(ax.ravel(), sel):
        a.imshow(im, interpolation="nearest"); a.set_title(f"{f[6:]}/{f[4:6]}/{f[:4]}", fontsize=11)
    fig.suptitle(f"NISAR GCOV banda L - {nombre} - orbita {k[0]} {k[1]} - rojo HH, verde HV, azul HH/HV - "
                 f"NEGRO = agua - imagen del radar, sin validar", fontsize=11)
    plt.tight_layout(); fig.savefig(os.path.join(OUT, f"{nombre}_hoja.png"), dpi=80); plt.close(fig)
    print(f"  hoja: {os.path.join(OUT, nombre + '_hoja.png')}", flush=True)

if __name__ == "__main__":
    if "--prueba" in sys.argv:
        prueba(); sys.exit(0)
    print("Cargando librerias (1-3 min en Windows, no cortar)...", flush=True)
    import earthaccess, h5py, requests, pyproj, matplotlib
    print("Librerias cargadas.", flush=True)
    earthaccess.login(persist=True)
    os.makedirs(OUT, exist_ok=True)
    if "--fechas" in sys.argv:
        FECHAS = sys.argv[sys.argv.index("--fechas") + 1].split(",")
    pedidas = [a for a in sys.argv[1:] if a in ZONAS] or list(ZONAS)
    for z in pedidas:
        zona(z, ZONAS[z])
    print("\nListo.")
