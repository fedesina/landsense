"""
sismo_la_guaira_danio_validacion.py -- LandSense
Valida el mapa de dano por coherencia (src/sismo_la_guaira_danio.py) contra PUNTOS DE DANO VERIFICADOS EN EL TERRENO.

Referencia: HOT (Humanitarian OpenStreetMap Team), HDX dataset
  "Venezuela - M 7.5 Earthquake - Damage Assessment Area", archivo
  chatmap_field_validated_damage_points.geojson (415 puntos: complete / significant / minimal).
  Licencia CC BY-NC-SA 4.0 (uso no comercial, atribucion, compartir igual).
  Es independiente del radar: son puntos constatados en campo (ChatMap).

Que mide (y que NO):
  - SENSIBILIDAD: que fraccion del dano verificado cae sobre (o cerca de) un pixel marcado.
  - Comparada contra: (a) un pixel estable al azar de la misma zona, (b) el mapa de CONTROL.
  - NO mide precision (cuanto de lo marcado es dano): los puntos de campo son solo lugares
    danados, no hay puntos "sanos". Medir precision requiere un inventario de edificios intactos.

Resultado (track 126, punto exacto): derrumbe completo 72%, dano importante 30%,
dano menor 14% | azar 2% | control 0%. A 40 m: 95% / 65% / 24% | azar 8% | control 2%.

Uso:
    python src/sismo_la_guaira_danio_validacion.py
Requiere: haber corrido src/sismo_la_guaira_danio.py (dpm_061.npz, dpm_126.npz). Necesita scipy.
"""
import os, json
import numpy as np
import requests
from pyproj import Transformer
from scipy.ndimage import binary_dilation

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(AQUI, "..", "data", "processed", "la_guaira", "danio")
URL_CAMPO = ("https://data.humdata.org/dataset/96d4c883-4e65-44fe-80b8-9ca709c91ca6/resource/"
             "89df1b21-e40e-417f-9fe2-118e98e77140/download/damage_points.geojson")
ARCH = os.path.join(DIR, "hot_campo_damage_points.geojson")
LOG = open(os.path.join(DIR, "validacion_campo.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

if not os.path.exists(ARCH):
    r = requests.get(URL_CAMPO, timeout=120); r.raise_for_status()
    open(ARCH, "wb").write(r.content)
pts = json.load(open(ARCH, encoding="utf-8"))["features"]
lon = np.array([q["geometry"]["coordinates"][0] for q in pts])
lat = np.array([q["geometry"]["coordinates"][1] for q in pts])
cat = np.array([q["properties"]["damaged"] for q in pts])
p(f"Puntos de campo HOT: {len(pts)} | " + ", ".join(f"{k}: {(cat == k).sum()}" for k in ["complete", "significant", "minimal"]))

rng = np.random.default_rng(0)
for t in ["126", "061"]:
    z = np.load(os.path.join(DIR, f"dpm_{t}.npz"))
    x, y, epsg = z["x"], z["y"], int(z["epsg"])
    marc, est, u = z["marcado"], z["estable"], float(z["umbral"])
    ctrl = est & (np.nan_to_num(z["z_ctrl"], nan=-9) >= u)
    X, Y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    dx, dy = x[1] - x[0], y[1] - y[0]
    ix = np.round((X - x[0]) / dx).astype(int); iy = np.round((Y - y[0]) / dy).astype(int)
    den = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
    info = np.zeros(len(pts), bool)
    info[den] = binary_dilation(est, iterations=2)[iy[den], ix[den]]
    ok = den & info
    p(f"\n=== Track {t} (pixel {abs(dx):.0f} m, umbral z >= {u:.2f}) | puntos en el recorte {den.sum()}, con informacion {ok.sum()}")
    r0, r1, c0, c1 = iy[den].min(), iy[den].max(), ix[den].min(), ix[den].max()
    cand = np.argwhere(est[r0:r1 + 1, c0:c1 + 1])
    sel = cand[rng.choice(len(cand), min(5000, len(cand)), replace=False)]
    ry, rx = sel[:, 0] + r0, sel[:, 1] + c0
    for R in [0, 2, 5]:                                   # 0, 40 y 100 m (pixeles de 20 m)
        m = binary_dilation(marc, iterations=R) if R else marc
        c = binary_dilation(ctrl, iterations=R) if R else ctrl
        hit = np.zeros(len(pts), bool); hit[ok] = m[iy[ok], ix[ok]]
        hitc = np.zeros(len(pts), bool); hitc[ok] = c[iy[ok], ix[ok]]
        fila = " | ".join(f"{k} {hit[ok & (cat == k)].mean():.0%} (n={int((ok & (cat == k)).sum())})"
                          for k in ["complete", "significant", "minimal"])
        p(f"  radio {R * 20:>3} m: EVENTO {fila} | todos {hit[ok].mean():.0%} || azar {m[ry, rx].mean():.0%} || CONTROL {hitc[ok].mean():.0%}")
p("\nLimites: mide SENSIBILIDAD, no precision; el azar incluye pixeles estables no urbanos; pixel de 20 m = manzanas, no edificios.")
LOG.close()
