"""
sismo_la_guaira_danio_export_visor.py -- LandSense
Lleva al visor el mapa de dano por coherencia (src/sismo_la_guaira_danio.py) como "ZONAS DE DERRUMBE PROBABLE"
con NIVELES DE CONFIANZA, mas los puntos de dano VERIFICADOS EN CAMPO (HOT).

Por que zonas y no pixeles: la validacion (src/sismo_la_guaira_danio_validacion.py) mostro que el metodo encuentra casi
todos los derrumbes completos pero NO clasifica edificio por edificio.
La escala honesta es el barrio: celdas de ~250 m (0.00225 grados).

Criterios (fijados de antemano, NO ajustados mirando los puntos de campo):
  1. Solo zonas pobladas: densidad WorldPop 2020 >= 300 hab/km2 = umbral de "cluster urbano" del
     Degree of Urbanisation (ONU, Comision de Estadistica 2020; GHSL/Eurostat). Simplificacion:
     solo el umbral de densidad, sin reglas de contiguidad ni poblacion minima total.
  2. Por orbita y celda: fraccion de pixeles estables marcados; umbral = p99 de esa fraccion en el
     CONTROL de esa orbita (par sin sismo), en celdas urbanas con >= 20 pixeles estables.
  3. Confianza: ALTA = la marcan las DOS orbitas independientes (061 ascendente y 126 descendente);
     MEDIA = una sola orbita. Lo mismo se aplica al control para medir falsa alarma.

Resultado: confirmado por dos orbitas: 97 zonas del sismo vs 1 del control; contiene
248 de 328 derrumbes completos verificados en campo (control: 0). Solo 126: 174 vs 103.
Sensibilidad con 1500 hab/km2 ("centro urbano"): se probo aparte y no se adopto (se vio despues del resultado).

Puntos de campo: HOT/HDX, CC BY-NC-SA 4.0.
ORDEN: correr despues de src/sismo_la_guaira_export_visor.py y src/sismo_la_guaira_poblacion.py (agrega el bloque "danio" a punto.json).
Requiere: src/sismo_la_guaira_danio.py corrido (dpm_061.npz, dpm_126.npz) y src/sismo_la_guaira_danio_validacion.py corrido (baja los puntos de HOT).
Uso:
    python src/sismo_la_guaira_danio_export_visor.py
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
DIR = os.path.join(AQUI, "..", "data", "processed", "la_guaira", "danio")
PUNTO = os.path.join(AQUI, "..", "web", "puntos", "la_guaira", "punto.json")
SALIDA = os.path.dirname(PUNTO)
TRACKS = ["126", "061"]
C = 0.00225                      # ~250 m
W, N, E, S = -67.13, 10.66, -66.69, 10.43   # grilla comun lon/lat que cubre el recorte de src/sismo_la_guaira_danio.py
NX, NY = int(round((E - W) / C)), int(round((N - S) / C))
MIN_PIX = 20
DENS_URBANA = 300
WP_URL = "https://worldpop.arcgis.com/arcgis/rest/services/WorldPop_Population_Density_1km/ImageServer/exportImage"

assert os.path.exists(PUNTO), "falta punto.json: correr src/sismo_la_guaira_export_visor.py (y 28)"
assert os.path.exists(os.path.join(DIR, "hot_campo_damage_points.geojson")), "faltan puntos HOT: correr src/sismo_la_guaira_danio_validacion.py"
punto = json.load(open(PUNTO, encoding="utf-8"))

# Densidad WorldPop 2020 en el centro de cada celda (vecino mas cercano)
r = requests.get(WP_URL, params=dict(bbox=f"{W},{S},{E},{N}", bboxSR=4326, imageSR=4326, size=f"{NX},{NY}",
                 format="tiff", pixelType="F32", f="image", interpolation="RSP_NearestNeighbor",
                 time=1577836800000), timeout=120)
r.raise_for_status()
with rasterio.open(io.BytesIO(r.content)) as ds:
    dens = ds.read(1).astype(float)
assert dens.shape == (NY, NX), "WorldPop devolvio otra grilla"
dens[(dens < 0) | (dens > 1e6) | ~np.isfinite(dens)] = 0
urbana = dens >= DENS_URBANA

def por_orbita(t):
    z = np.load(os.path.join(DIR, f"dpm_{t}.npz"))
    x, y, epsg = z["x"], z["y"], int(z["epsg"])
    est, marc = z["estable"], z["marcado"]
    ctrl = est & (np.nan_to_num(z["z_ctrl"], nan=-9) >= float(z["umbral"]))
    X, Y = np.meshgrid(x, y)
    lo, la = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(X[est], Y[est])
    c = ((lo - W) / C).astype(int); rr = ((N - la) / C).astype(int)
    ok = (c >= 0) & (c < NX) & (rr >= 0) & (rr < NY)
    idx = rr[ok] * NX + c[ok]
    npx = np.bincount(idx, minlength=NX * NY).reshape(NY, NX)
    fe = np.bincount(idx, weights=marc[est][ok], minlength=NX * NY).reshape(NY, NX) / np.maximum(npx, 1)
    fc = np.bincount(idx, weights=ctrl[est][ok], minlength=NX * NY).reshape(NY, NX) / np.maximum(npx, 1)
    val = urbana & (npx >= MIN_PIX)
    u = float(np.percentile(fc[val], 99))
    print(f"Track {t}: celdas urbanas con dato {val.sum()} | umbral (p99 control) {u:.1%} | "
          f"evento {(val & (fe > u)).sum()} | control {(val & (fc > u)).sum()}")
    return dict(val=val, ev=val & (fe > u), ct=val & (fc > u), fe=fe, u=u)

O = {t: por_orbita(t) for t in TRACKS}
a, b = O[TRACKS[0]], O[TRACKS[1]]
alta = a["ev"] & b["ev"]
media = (a["ev"] | b["ev"]) & ~alta
ct_alta = a["ct"] & b["ct"]
print(f"Confianza ALTA (dos orbitas): {alta.sum()} zonas | control con dos orbitas: {ct_alta.sum()}")
print(f"Confianza MEDIA (una orbita): {media.sum()} zonas")

campo = json.load(open(os.path.join(DIR, "hot_campo_damage_points.geojson"), encoding="utf-8"))["features"]
pts = [dict(lon=round(q["geometry"]["coordinates"][0], 5), lat=round(q["geometry"]["coordinates"][1], 5),
            cat=q["properties"]["damaged"]) for q in campo]
def en(Z, q):
    c = int((q["lon"] - W) / C); rr = int((N - q["lat"]) / C)
    return 0 <= c < NX and 0 <= rr < NY and bool(Z[rr, c])
comp = [q for q in pts if q["cat"] == "complete"]
h_alta = sum(en(alta, q) for q in comp); h_any = sum(en(alta | media, q) for q in comp)
h_ct = sum(en(ct_alta, q) for q in comp)
print(f"Derrumbes completos verificados (n={len(comp)}): en zonas ALTA {h_alta} | en ALTA o MEDIA {h_any} | en control ALTA {h_ct}")

rgba = np.zeros((NY, NX, 4))
rgba[media] = [0.99, 0.55, 0.24, 0.65]      # naranja: una orbita
rgba[alta] = [0.65, 0.06, 0.08, 0.9]        # rojo oscuro: dos orbitas
plt.imsave(os.path.join(SALIDA, "danio_zonas.png"), rgba)

punto["danio"] = dict(
    archivo="danio_zonas.png",
    bounds=dict(west=W, east=W + NX * C, south=N - NY * C, north=N),
    celda_m=250,
    densidad_minima=DENS_URBANA,
    zonas_alta=int(alta.sum()), zonas_media=int(media.sum()), control_alta=int(ct_alta.sum()),
    texto=("Zonas donde el radar dejó de reconocer el terreno después del sismo, la huella típica de "
           "derrumbes. Marca barrios donde es probable encontrarlos, no edificios uno por uno."),
    validacion=(f"Confirmadas por dos satélites: {int(alta.sum())} zonas; la misma prueba sin sismo marcó "
                f"{int(ct_alta.sum())}. Contienen {h_alta} de los {len(comp)} derrumbes completos verificados en el "
                "terreno por voluntarios (HOT)."),
    limites=[
        "Marca zonas de unos 250 m, no edificios: aproximadamente 7 de cada 10 edificios en zonas marcadas no "
        "muestran daño visible.",
        "Solo zonas pobladas (300 habitantes por km² o más, definición de área urbana de la ONU) y con terreno "
        "estable antes del sismo; no informa en vegetación ni agua.",
        "Naranja = visto por un solo satélite: confianza media, puede ser ruido.",
        "Ve derrumbes, no daño parcial: los daños parciales verificados en Caracas no aparecen.",
        "Otras cosas también hacen caer la señal: escombros, obras, movimiento en calles.",
    ],
    fuente=("NISAR GUNW provisional, coherencia 20 m, órbitas 061 (ascendente) y 126 (descendente); línea base "
            "de 4 pares previos y control con un par sin sismo. Población: WorldPop 2020."),
    campo=dict(puntos=pts, completos=len(comp), completos_en_alta=int(h_alta),
               fuente="Daño verificado en campo: Humanitarian OpenStreetMap Team (HOT) / ChatMap, vía HDX. "
                      "Licencia CC BY-NC-SA 4.0."),
)
json.dump(punto, open(PUNTO, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Capa: {os.path.join(SALIDA, 'danio_zonas.png')} | bloque 'danio' agregado a punto.json")
