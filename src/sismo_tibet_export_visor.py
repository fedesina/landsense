"""
sismo_tibet_export_visor.py -- LandSense
Escena del visor para el sismo del TIBET M7,0 (us6000u43g, 22-oct-2026), igual criterio que Panama (src/sismo_panama_export_visor.py):
SIN flechas ni este-oeste/vertical. Solo lo que se sostiene tras la auditoria:
  - Capas = desplazamiento en LINEA DE VISTA (LOS) de las dos orbitas ASCENDENTES, que coinciden (r 0,93-0,96):
      112 A f018 (19 -> 31 oct)  y  040 A f017+f018 (14 -> 26 oct; solo cubre el borde oeste de la zona).
  - Pares que cruzan de la coleccion Urgent Response (src/sismo_generico.py --con-ur); controles de PROVISIONAL.
  - La descendente 163 D no tiene datos validos en la zona: no hay descomposicion.
  - El hueco sin dato junto a la ruptura es sobre todo zona SIN DESENROLLAR (src/sismo_tibet_chequeo_hueco.py, caso B): se muestra
    transparente y los valores de su borde no se citan.
Entrada: data/processed/sismos/us6000u43g/pares_los.npz (lo guarda src/sismo_generico.py).
Salida:  web/puntos/tibet/{los_112A.png, los_040A.png, punto.json}  + agrega "tibet" a web/puntos/index.json
Requiere internet (epicentro y falla finita del USGS en vivo, como src/sismo_panama_export_visor.py).
Uso:  python src/sismo_tibet_export_visor.py
"""
import os, json
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

AQUI = os.path.dirname(os.path.abspath(__file__))
EVENTO = "us6000u43g"
ENTRADA = os.path.join(AQUI, "..", "data", "processed", "sismos", EVENTO, "pares_los.npz")
WEB = os.path.join(AQUI, "..", "web", "puntos")
SALIDA = os.path.join(WEB, "tibet"); os.makedirs(SALIDA, exist_ok=True)
LIM_CM = 50.0          # escala fija +-50 cm (junto al hueco pasa de eso: satura, se aclara en la ficha)
PASO = 0.0009          # ~85-100 m (el dato es de 80 m)
CAPAS = [("los_112A", ["112A_018_cruza"], "19 de octubre", "31 de octubre"),
         ("los_040A", ["040A_018_cruza", "040A_017_cruza"], "14 de octubre", "26 de octubre")]
USGS_EV = "https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={}&format=geojson"

assert os.path.exists(ENTRADA), "falta pares_los.npz: correr src/sismo_generico.py us6000u43g --con-ur primero"
z = np.load(ENTRADA, allow_pickle=True)

def datos(clave):
    return z[clave + "__los"], z[clave + "__x"], z[clave + "__y"], int(z[clave + "__epsg"])

ext = []
for _, claves, *_ in CAPAS:
    for k in claves:
        L, x, y, ep = datos(k)
        f = np.where(np.isfinite(L).any(1))[0]; c = np.where(np.isfinite(L).any(0))[0]
        X, Y = np.meshgrid(x[[c.min(), c.max()]], y[[f.min(), f.max()]])
        lo, la = Transformer.from_crs(f"EPSG:{ep}", "EPSG:4326", always_xy=True).transform(X, Y); ext.append((lo, la))
oeste = min(e[0].min() for e in ext); este = max(e[0].max() for e in ext)
sur = min(e[1].min() for e in ext); norte = max(e[1].max() for e in ext)
lons = np.arange(oeste, este, PASO) + PASO / 2; lats = np.arange(norte, sur, -PASO) - PASO / 2
LON, LAT = np.meshgrid(lons, lats)
bounds = dict(west=float(lons[0] - PASO / 2), east=float(lons[-1] + PASO / 2),
              north=float(lats[0] + PASO / 2), south=float(lats[-1] - PASO / 2))

def reproyectar(clave):
    L, x, y, ep = datos(clave)
    XU, YU = Transformer.from_crs("EPSG:4326", f"EPSG:{ep}", always_xy=True).transform(LON, LAT)
    ix = np.round((XU - x[0]) / (x[1] - x[0])).astype(int); iy = np.round((YU - y[0]) / (y[1] - y[0])).astype(int)
    ok = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
    out = np.full(LON.shape, np.nan, np.float32); out[ok] = L[iy[ok], ix[ok]]
    po, pr = np.nanpercentile(L, [1, 50, 99]), np.nanpercentile(out, [1, 50, 99])
    print(f"  {clave}: original p1/p50/p99 {po.round(1)} | reproyectado {pr.round(1)}")
    assert np.all(np.abs(po - pr) < 5.0), f"la reproyeccion de {clave} cambio el dato"
    return out

capas = []
for cid, claves, f0, f1 in CAPAS:
    rs = [reproyectar(k) for k in claves]
    if len(rs) > 1:   # frames de la MISMA pasada: corrimiento constante por desenrollado propio; se resta y se exige coincidencia
        a, b = rs; m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 1000:
            off = float(np.median(a[m] - b[m])); b = b + off; rs[1] = b
            dif = np.abs(a[m] - b[m])
            print(f"  superposicion {claves}: {m.sum()} px, corrimiento {off:.1f} cm; tras restarlo |dif| p50 {np.median(dif):.1f} / p95 {np.percentile(dif, 95):.1f} cm")
            assert np.percentile(dif, 95) < 10, "los frames de la misma pasada no coinciden ni tras restar el corrimiento: revisar"
        else:
            print(f"  superposicion {claves}: {m.sum()} px (casi no se tocan): se muestran juntos sin ajustar")
        campo = np.nanmean(np.stack(rs), 0)
    else:
        campo = rs[0]
    rgba = plt.get_cmap("RdBu_r")(np.clip((campo + LIM_CM) / (2 * LIM_CM), 0, 1))
    rgba[..., 3] = np.where(np.isfinite(campo), 1.0, 0.0)
    plt.imsave(os.path.join(SALIDA, cid + ".png"), rgba)
    v = campo[np.isfinite(campo)]
    orb = cid.split("_")[1]
    capas.append(dict(id=cid, nombre=f"Órbita {orb[:3]} ({f0} → {f1})", izquierda="se alejó", derecha="se acercó",
                      texto_izq="el suelo se alejó del satélite (hacia el oeste o hacia abajo)",
                      texto_der="el suelo se acercó al satélite (hacia el este o hacia arriba)",
                      texto_neutro="blanco: casi no se movió", archivo=cid + ".png", validos=int(v.size),
                      p1=round(float(np.percentile(v, 1)), 1), p99=round(float(np.percentile(v, 99)), 1)))
    print(f"  {cid}.png: {v.size} px | p1 {capas[-1]['p1']} / p99 {capas[-1]['p99']} cm")

# ---------- USGS en vivo ----------
d = requests.get(USGS_EV.format(EVENTO), timeout=60).json()
lo, la, prof = d["geometry"]["coordinates"]
epicentros = [dict(id=EVENTO, mag=d["properties"]["mag"], lugar=d["properties"]["place"], lon=lo, lat=la, prof_km=prof,
                   dentro=bool(bounds["west"] <= lo <= bounds["east"] and bounds["south"] <= la <= bounds["north"]))]
print(f"USGS {EVENTO}: M{d['properties']['mag']} {d['properties']['place']} ({lo:.3f}, {la:.3f})")
falla = None
try:
    ff = d["properties"]["products"]["finite-fault"][0]
    ffm = requests.get(ff["contents"]["FFM.geojson"]["url"], timeout=60).json()
    filas = []
    for f in ffm["features"]:
        c = np.array(f["geometry"]["coordinates"][0]); zmin = c[:, 2].min()
        filas.append((zmin, [tuple(p[:2]) for p in c if p[2] <= zmin + 1]))
    z0 = min(zz for zz, _ in filas)
    pts = sorted({(round(float(p[0]), 3), round(float(p[1]), 3)) for zz, ps in filas if zz <= z0 + 1000 for p in ps},
                 key=lambda p: (p[1], p[0]))
    if len(pts) >= 2:
        falla = dict(nombre="Falla (modelo USGS, trazo aproximado)", puntos=pts,
                     fuente=f"Falla: borde superior del modelo de falla finita del USGS para {EVENTO} (versión {ff['properties'].get('version')}).",
                     nota="Borde superior del plano del modelo de falla finita; no es el trazo exacto en superficie. Ese modelo usó también datos de radar, así que no es una comparación independiente.")
        print(f"Traza USGS (falla finita v{ff['properties'].get('version')}): {len(pts)} puntos")
except Exception as ex:
    print(f"Sin falla finita utilizable ({type(ex).__name__}): la escena va sin trazo de falla.")

P = dict(
    id="tibet", nombre="Terremoto del Tíbet", lugar="Nagqu, Tíbet (China)", tipo="antes_despues",
    titular="NISAR vio el suelo moverse decenas de centímetros alrededor del epicentro",
    que_es="Este mapa no es una foto: compara dos pasadas del radar NISAR, una antes y otra después del sismo, y muestra cuánto se acercó o se alejó cada punto del suelo respecto del satélite. Lo vieron dos órbitas distintas, que coinciden donde se superponen.",
    fecha_antes="19 de octubre de 2026", fecha_despues="31 de octubre de 2026",
    evento="Sismo M7,0 del 22 de octubre de 2026 a 18 km de profundidad; falla normal con componente lateral según USGS. Prensa: al menos 16 muertos y unas 339 casas derrumbadas.",
    satelite="NISAR (NASA-ISRO), banda L, productos GUNW de respuesta urgente (UR)",
    pares="Ascendente 112 (19 oct - 31 oct, frame 018) y ascendente 040 (14 oct - 26 oct, frames 017 y 018). Controles: el par anterior de cada órbita (colección provisional).",
    estado="Visto, falta confirmar", color="amarillo",
    estado_detalle="Dos órbitas ascendentes distintas ven el mismo movimiento donde se superponen (r = 0,93 y 0,96), y en los pares de control (sin sismo) no aparece nada. No hay pasada descendente con datos, así que no podemos separar cuánto fue hundimiento y cuánto movimiento de costado. La confirmación independiente queda pendiente.",
    resultado="src/sismo_generico.py --con-ur: C1 asc 23,9× (112 A) y 4,9-5,5× (040 A); desc sin datos (163 D) → C1 no pasa por regla. C2 PASA (040 A vs 112 A r = 0,93 / 0,96). C3 y C4 no evaluables (sin descendente). Sin errores de desenrollado (0,47 saltos de λ/2 por 10.000 vs 22,6 en el par fallado de Panamá). Hueco junto a la ruptura: 64 % sin desenrollar y 36 % de baja coherencia (src/sismo_tibet_chequeo_hueco.py).",
    limites=["Mide solo en la línea de vista del satélite: azul puede ser hacia el oeste o hacia abajo; sin una pasada descendente no se separa.",
             "La pasada descendente (163 D) no tiene datos válidos en la zona, ni antes ni después del sismo.",
             "Junto a la ruptura hay una zona sin dato: ahí el movimiento cambia tan de golpe que el cálculo no pudo seguirlo (y en parte el radar no reconoce el suelo). Los valores pegados a esa zona no se citan.",
             "La órbita 040 solo cubre el borde oeste de la zona, lejos del epicentro.",
             "Los pares que cruzan el sismo son de la colección de respuesta urgente de NASA; los controles, de la provisional (mismo procesamiento).",
             "Valores relativos dentro de la imagen; no se predicen sismos: se mide lo que ya ocurrió."],
    bounds=bounds, escala_cm=LIM_CM, capas=capas,
    camara=dict(lon=lo, lat=la, radio_m=50000, distancia_m=170000),
    epicentros=epicentros,
)
if falla:
    P["falla"] = falla
json.dump(P, open(os.path.join(SALIDA, "punto.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ij = os.path.join(WEB, "index.json"); I = json.load(open(ij, encoding="utf-8"))
if "tibet" not in I["puntos"]:
    I["puntos"].append("tibet"); json.dump(I, open(ij, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Listo:", os.path.join(SALIDA, "punto.json"))
