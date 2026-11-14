"""
sismo_la_guaira_poblacion.py -- LandSense
Cruza el desplazamiento co-sismico VERIFICADO (src/sismo_la_guaira_descomposicion.py) con la poblacion WorldPop,
en las celdas NATIVAS de WorldPop (~920 m), sin inventar detalle.

Decisiones de diseno (para no inflar la cifra):
  - usa solo el mapa de movimiento verificado;
  - pide el ano 2020 explicito: el servicio es multi-anual (2000-2020) y sin 'time' devuelve
    otro ano (medido: 3.74 M vs 4.40 M personas en la misma caja);
  - muestra aparte la gente donde el radar no tiene dato (sin subestimacion silenciosa);
  - no estira celdas de 1 km a 80 m (sin falso detalle).

Lo que se puede afirmar con esto: cuanta gente vive (estimacion WorldPop 2020) en celdas
donde el suelo se movio mas que el umbral de confianza. NO es "gente afectada":
el dano lo produce el sacudon, no el desplazamiento permanente.

ORDEN: correr DESPUES de src/sismo_la_guaira_export_visor.py (ese script reescribe punto.json y borra este bloque).
Probado con el dato real.

Salidas:
  ../data/processed/la_guaira/poblacion.txt
  ../web/puntos/la_guaira/poblacion.png   (capa para el visor)
  agrega el bloque "poblacion" a ../web/puntos/la_guaira/punto.json
Uso:
    python src/sismo_la_guaira_poblacion.py
"""
import os, io, json
import numpy as np
import requests
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

AQUI = os.path.dirname(os.path.abspath(__file__))
ENTRADA = os.path.join(AQUI, "..", "data", "processed", "la_guaira", "EU_coseismico.npz")
PUNTO = os.path.join(AQUI, "..", "web", "puntos", "la_guaira", "punto.json")
SALIDA_WEB = os.path.dirname(PUNTO)
LOGF = os.path.join(AQUI, "..", "data", "processed", "la_guaira", "poblacion.txt")
LOG = open(LOGF, "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

WP_URL = ("https://worldpop.arcgis.com/arcgis/rest/services/"
          "WorldPop_Population_Density_1km/ImageServer/exportImage")
WP_ANIO_MS = 1577836800000        # 2020-01-01: ultimo ano del servicio (verificado en sus metadatos)
PX = 0.00833333329999927          # tamano de celda nativo (metadatos del servicio)
COBERTURA_MIN = 0.30              # fraccion minima de la celda con dato de radar
TAM_PIX_RADAR_M = 80.0            # posting del GUNW

assert os.path.exists(ENTRADA), "falta EU_coseismico.npz: correr src/sismo_la_guaira_descomposicion.py"
assert os.path.exists(PUNTO), "falta punto.json: correr src/sismo_la_guaira_export_visor.py"
punto = json.load(open(PUNTO, encoding="utf-8"))
UMBRAL = float(punto["flechas"]["minimo_cm"])      # el mismo umbral de confianza que las flechas
p(f"Umbral de confianza (del punto.json): {UMBRAL:.1f} cm")

# ---------- 1) Desplazamiento este-oeste con coordenadas geograficas ----------
z = np.load(ENTRADA)
E, x, y, epsg = z["E_cm"], z["x"], z["y"], int(z["epsg"])
X, Y = np.meshgrid(x, y)
lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(X, Y)
ok = np.isfinite(E)
lon_v, lat_v, E_v = lon[ok], lat[ok], E[ok]

# ---------- 2) Grilla nativa de WorldPop que cubre el dato ----------
w = np.floor(lon_v.min() / PX) * PX; e = np.ceil(lon_v.max() / PX) * PX
s = np.floor(lat_v.min() / PX) * PX; n = np.ceil(lat_v.max() / PX) * PX
nx = int(round((e - w) / PX)); ny = int(round((n - s) / PX))
params = dict(bbox=f"{w},{s},{e},{n}", bboxSR=4326, imageSR=4326, size=f"{nx},{ny}",
              format="tiff", pixelType="F32", f="image",
              interpolation="RSP_NearestNeighbor", time=WP_ANIO_MS)
r = requests.get(WP_URL, params=params, timeout=120); r.raise_for_status()
with rasterio.open(io.BytesIO(r.content)) as ds:
    dens = ds.read(1).astype(float)
    assert (ds.width, ds.height) == (nx, ny), "WorldPop devolvio otra grilla: revisar"
    assert abs(ds.transform[0] - PX) < 1e-9, "tamano de celda distinto al nativo: revisar"
dens[(dens < 0) | (dens > 1e6) | ~np.isfinite(dens)] = np.nan
lat_c = n - (np.arange(ny) + 0.5) * PX                 # fila 0 = norte
lon_c = w + (np.arange(nx) + 0.5) * PX
area_km2 = (PX * 111.32) * (PX * 111.32 * np.cos(np.radians(lat_c)))[:, None] * np.ones((1, nx))
pob = dens * area_km2
p(f"WorldPop 2020: {nx}x{ny} celdas de ~{PX*111.32*1000:.0f} m | poblacion total en la caja {np.nansum(pob):,.0f}")

# ---------- 3) Por celda: cobertura de radar y mediana de E ----------
ci = np.floor((lon_v - w) / PX).astype(int); ri = np.floor((n - lat_v) / PX).astype(int)
dentro = (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny)
ci, ri, Ev = ci[dentro], ri[dentro], E_v[dentro]
idx = ri * nx + ci
cuenta = np.bincount(idx, minlength=nx * ny).reshape(ny, nx)
orden = np.argsort(idx); idx_o, E_o = idx[orden], Ev[orden]
mediana = np.full(nx * ny, np.nan)
cortes = np.flatnonzero(np.diff(idx_o)) + 1
for grupo_idx, grupo_E in zip(np.split(idx_o, cortes), np.split(E_o, cortes)):
    mediana[grupo_idx[0]] = np.median(grupo_E)
mediana = mediana.reshape(ny, nx)
esperados = area_km2 * 1e6 / TAM_PIX_RADAR_M ** 2
cobertura = cuenta / esperados

# Celdas dentro de la zona medida: las que tocan el rectangulo del dato (bounds del visor)
b = punto["bounds"]
LONc, LATc = np.meshgrid(lon_c, lat_c)
en_zona = (LONc >= b["west"]) & (LONc <= b["east"]) & (LATc >= b["south"]) & (LATc <= b["north"]) & np.isfinite(pob)

con_dato = en_zona & (cobertura >= COBERTURA_MIN)
fuerte = con_dato & (np.abs(mediana) >= UMBRAL)
menor = con_dato & (np.abs(mediana) < UMBRAL)
sin_dato = en_zona & ~con_dato

def tot(m): return float(np.nansum(pob[m]))
p("\n--- RESULTADO (estimacion WorldPop 2020, celdas ~920 m) ---")
p(f"Celdas en la zona medida: {int(en_zona.sum())} | con dato suficiente (>= {COBERTURA_MIN:.0%} de la celda): {int(con_dato.sum())}")
p(f"Personas en celdas con desplazamiento >= {UMBRAL:.0f} cm (zona de mayor deformacion): {tot(fuerte):,.0f}")
p(f"Personas en celdas con desplazamiento menor al umbral: {tot(menor):,.0f}")
p(f"Personas en celdas SIN dato suficiente de radar (no se sabe): {tot(sin_dato):,.0f}")
escalones = {}
for c in [UMBRAL, 25, 40]:
    m = con_dato & (np.abs(mediana) >= c)
    escalones[f"{c:.0f}"] = round(tot(m))
    p(f"  >= {c:.0f} cm: {tot(m):,.0f} personas en {int(m.sum())} celdas")
top = np.argsort(np.where(fuerte, pob, -1).ravel())[::-1][:5]
p("Celdas con mas gente dentro de la zona de mayor deformacion:")
for k in top:
    rr, cc_ = divmod(k, nx)
    if fuerte[rr, cc_]:
        p(f"  ({lon_c[cc_]:.3f}, {lat_c[rr]:.3f}): {pob[rr, cc_]:,.0f} personas, E mediana {mediana[rr, cc_]:+.0f} cm, cobertura {cobertura[rr, cc_]:.0%}")

# ---------- 4) Capa para el visor: gente en la zona de mayor deformacion ----------
vmax = float(np.nanpercentile(pob[fuerte], 95)) if fuerte.any() else 1.0
norm = np.clip(pob / vmax, 0, 1)
rgba = plt.get_cmap("YlOrRd")(0.25 + 0.75 * norm)
rgba[..., 3] = np.where(fuerte, 0.85, 0.0)
rgba[sin_dato, :3] = 0.5; rgba[sin_dato, 3] = 0.35       # gris: sin dato suficiente
plt.imsave(os.path.join(SALIDA_WEB, "poblacion.png"), rgba)

punto["poblacion"] = dict(
    archivo="poblacion.png",
    bounds=dict(west=float(w), east=float(e), south=float(s), north=float(n)),
    fuente="WorldPop 2020, densidad 1 km (doi 10.5258/SOTON/WP00674), servicio ArcGIS de WorldPop/Esri",
    anio=2020,
    umbral_cm=UMBRAL,
    personas_mayor_deformacion=round(tot(fuerte)),
    personas_deformacion_menor=round(tot(menor)),
    personas_sin_dato=round(tot(sin_dato)),
    escalones=escalones,
    escala_max_personas_celda=round(vmax),
    texto=(f"Unas {round(tot(fuerte), -3):,.0f} personas vivían (estimación 2020) donde el suelo se corrió "
           f"más de {UMBRAL:.0f} cm.").replace(",", "."),
    limites=[
        "Población estimada para 2020 (último año disponible); no es la del día del sismo.",
        "Es exposición cerca de la ruptura, no gente afectada: el daño lo produce el sacudón.",
        "Celdas de ~1 km: indica zonas, no manzanas.",
        f"En gris, celdas donde el radar no tiene dato suficiente: ahí no se sabe cuánto se movió el suelo.",
    ],
)
json.dump(punto, open(PUNTO, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
p(f"\nCapa: {os.path.join(SALIDA_WEB, 'poblacion.png')} | bloque 'poblacion' agregado a punto.json")
LOG.close()
