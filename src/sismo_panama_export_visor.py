"""
sismo_panama_export_visor.py -- LandSense
Escena del visor para el sismo de PANAMA M7,7 (us6000u18k, 9-oct-2026), estilo La Guaira (src/sismo_la_guaira_export_visor.py) pero
SIN flechas ni este-oeste/vertical: solo lo que se sostiene tras la auditoria.
  - Capas = desplazamiento en LINEA DE VISTA (LOS) de las dos orbitas DESCENDENTES, que coinciden (r 0,99):
      141 D (27-sep -> 9-oct): incluye M7,7 y M6,6.   040 D (2-oct -> 14-oct): incluye ademas M6,0 y M6,1.
  - La ascendente 076 A NO se usa (errores de desenrollado: saltos de lambda/2).
Entrada: data/processed/sismos/us6000u18k/pares_los.npz (lo guarda src/sismo_generico.py).
Salida:  web/puntos/panama/{los_141D.png, los_040D.png, punto.json}  + agrega "panama" a web/puntos/index.json
Requiere internet (epicentros y falla finita del USGS en vivo, como src/sismo_la_guaira_export_visor.py).
Uso:  python src/sismo_panama_export_visor.py
"""
import os, json
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

AQUI = os.path.dirname(os.path.abspath(__file__))
ENTRADA = os.path.join(AQUI, "..", "data", "processed", "sismos", "us6000u18k", "pares_los.npz")
WEB = os.path.join(AQUI, "..", "web", "puntos")
SALIDA = os.path.join(WEB, "panama"); os.makedirs(SALIDA, exist_ok=True)
LIM_CM = 100.0         # escala fija +-100 cm (cerca de la falla pasa de 1 m: satura, se aclara en la ficha)
PASO = 0.0008          # ~88 m (el dato es de 80 m)
# 040 D primero (un solo frame, sin unir); 141 D = el par mas corto despues del sismo (6 h), dos frames unidos
CAPAS = [("los_040D", ["040D_085_cruza"], "2 de octubre", "14 de octubre", "M7,7, M6,6, M6,0 y M6,1"),
         ("los_141D", ["141D_085_cruza", "141D_084_cruza"], "27 de septiembre", "9 de octubre", "M7,7 y M6,6")]
USGS_EV = "https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={}&format=geojson"

assert os.path.exists(ENTRADA), "falta pares_los.npz: correr src/sismo_generico.py us6000u18k primero"
z = np.load(ENTRADA, allow_pickle=True)

def a_lonlat(clave):
    L, x, y, ep = z[clave + "__los"], z[clave + "__x"], z[clave + "__y"], int(z[clave + "__epsg"])
    return L, x, y, ep

# extension comun (union de todos los frames usados), en grados
ext = []
for _, claves, *_ in CAPAS:
    for k in claves:
        L, x, y, ep = a_lonlat(k)
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
    L, x, y, ep = a_lonlat(clave)
    XU, YU = Transformer.from_crs("EPSG:4326", f"EPSG:{ep}", always_xy=True).transform(LON, LAT)
    ix = np.round((XU - x[0]) / (x[1] - x[0])).astype(int); iy = np.round((YU - y[0]) / (y[1] - y[0])).astype(int)
    ok = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
    out = np.full(LON.shape, np.nan, np.float32); out[ok] = L[iy[ok], ix[ok]]
    po, pr = np.nanpercentile(L, [1, 50, 99]), np.nanpercentile(out, [1, 50, 99])
    print(f"  {clave}: original p1/p50/p99 {po.round(1)} | reproyectado {pr.round(1)}")
    assert np.all(np.abs(po - pr) < 5.0), f"la reproyeccion de {clave} cambio el dato"
    return out

capas = []
for cid, claves, f0, f1, sis in CAPAS:
    rs = [reproyectar(k) for k in claves]
    if len(rs) > 1:   # frames de la MISMA pasada: deben coincidir donde se superponen
        a, b = rs; m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 1000:
            # cada frame se desenrolla con su propio cero -> corrimiento constante; se resta (mediana) y se
            # exige que lo que sobra sea chico (si no, no es solo un corrimiento: no se une)
            off = float(np.median(a[m] - b[m])); b = b + off; rs[1] = b
            dif = np.abs(a[m] - b[m])
            print(f"  superposicion {claves}: {m.sum()} px, corrimiento {off:.1f} cm; tras restarlo |dif| p50 {np.median(dif):.1f} / p95 {np.percentile(dif, 95):.1f} cm")
            assert np.percentile(dif, 95) < 10, "los frames de la misma pasada no coinciden ni tras restar el corrimiento: revisar"
        campo = np.nanmean(np.stack(rs), 0)
    else:
        campo = rs[0]
    rgba = plt.get_cmap("RdBu_r")(np.clip((campo + LIM_CM) / (2 * LIM_CM), 0, 1))
    rgba[..., 3] = np.where(np.isfinite(campo), 1.0, 0.0)
    plt.imsave(os.path.join(SALIDA, cid + ".png"), rgba)
    v = campo[np.isfinite(campo)]
    orb = cid.split("_")[1]
    capas.append(dict(id=cid, nombre=f"Órbita {orb[:3]} ({f0} → {f1})", izquierda="se alejó", derecha="se acercó",
                      texto_izq="el suelo se alejó del satélite", texto_der="el suelo se acercó al satélite (hacia el oeste o hacia arriba)",
                      texto_neutro="blanco: casi no se movió", archivo=cid + ".png", validos=int(v.size),
                      p1=round(float(np.percentile(v, 1)), 1), p99=round(float(np.percentile(v, 99)), 1), sismos=sis))
    print(f"  {cid}.png: {v.size} px | p1 {capas[-1]['p1']} / p99 {capas[-1]['p99']} cm")

# ---------- USGS en vivo ----------
epicentros = []
for ev in ["us6000u18k", "us6000u19w", "us6000u1cf", "us6000u1wn"]:
    d = requests.get(USGS_EV.format(ev), timeout=60).json()
    lo, la, prof = d["geometry"]["coordinates"]
    epicentros.append(dict(id=ev, mag=d["properties"]["mag"], lugar=d["properties"]["place"], lon=lo, lat=la, prof_km=prof,
                           dentro=bool(bounds["west"] <= lo <= bounds["east"] and bounds["south"] <= la <= bounds["north"])))
    print(f"USGS {ev}: M{d['properties']['mag']} {d['properties']['place']} ({lo:.3f}, {la:.3f}) dentro: {epicentros[-1]['dentro']}")
det = requests.get(USGS_EV.format("us6000u18k"), timeout=60).json()
ff = det["properties"]["products"]["finite-fault"][0]
ffm = requests.get(ff["contents"]["FFM.geojson"]["url"], timeout=60).json()
# El modelo de falla finita de USGS (version 3) tiene 3 segmentos con su fila de arriba a profundidades distintas (-84 m y -231 m): se toma la
# PRIMERA fila de subfallas (tope < minimo + 1 km), su borde superior, y se une segmento por segmento de oeste a este.
filas = []
for f in ffm["features"]:
    c = np.array(f["geometry"]["coordinates"][0]); zmin = c[:, 2].min()
    filas.append((zmin, [tuple(p[:2]) for p in c if p[2] <= zmin + 1]))
z0 = min(z for z, _ in filas)
grupos = {}
for zmin, pts in filas:
    if zmin <= z0 + 1000: grupos.setdefault(round(zmin), []).extend(pts)
traza = []
for zk in sorted(grupos, key=lambda k: min(p[0] for p in grupos[k])):
    traza += sorted({(round(float(p[0]), 3), round(float(p[1]), 3)) for p in grupos[zk]})
print(f"Traza USGS (falla finita v{ff['properties'].get('version')}): {len(traza)} puntos, lon {traza[0][0]}..{traza[-1][0]}")

P = dict(
    id="panama", nombre="Terremoto de Panamá", lugar="Península de Azuero, Panamá", tipo="antes_despues",
    titular="NISAR vio el suelo romperse a lo largo de la falla en la península de Azuero",
    que_es="Este mapa no es una foto: compara dos pasadas del radar NISAR, una antes y otra después del sismo, y muestra cuánto se acercó o se alejó cada punto del suelo respecto del satélite. Lo vieron dos órbitas distintas y, junto a la falla, coinciden.",
    fecha_antes="2 de octubre de 2026", fecha_despues="14 de octubre de 2026",
    evento="Sismo M7,7 del 9 de octubre de 2026 (y réplicas M6,6 el mismo día, M6,0 el 10 y M6,1 el 12), falla lateral izquierda según USGS",
    satelite="NISAR (NASA-ISRO), banda L, productos GUNW provisionales",
    pares="Descendente 040 (2 oct - 14 oct, frame 085) y descendente 141 (27 sep - 9 oct, frames 084 y 085 unidos restando su corrimiento de 75 cm; lo que sobra: 4 cm)",
    estado="Visto: coincide con USGS", color="amarillo",
    estado_detalle="Dos órbitas descendentes distintas ven la misma forma junto a la falla (r = 0,99). El sentido del movimiento (lateral izquierdo) y la dirección de la falla (~281° medido vs 288° de USGS) coinciden con el análisis sísmico de USGS. No damos cuánto se movió hacia el oeste ni hacia arriba: la única pasada ascendente tiene errores de procesamiento.",
    resultado="src/sismo_generico.py: C1 PASA (141 D 5,7-12×, 040 D 15,4×), C2 PASA (040 D vs 141 D r = 0,99 / 1,00), C3 NO CONFIABLE (076 A con saltos de λ/2 = 12,1 cm: 22,6 por 10.000 vecinos vs 0,3-1,0 en las descendentes). C4: bloque norte al oeste = lateral izquierdo; rumbo de la traza ≈ 281° (USGS: tensor 288°, falla finita 275-295°).",
    limites=["Mide solo en la línea de vista del satélite: rojo puede ser hacia el oeste o hacia arriba; sin una segunda geometría buena no se separa.",
             "La pasada ascendente (076 A) tiene errores de procesamiento (saltos de media longitud de onda) y no se usa.",
             "Cerca de la falla el suelo se movió más de un metro y la imagen se satura; justo sobre la falla no hay dato (el radar pierde coherencia).",
             "La pasada 040 D incluye también las réplicas del 10 y 12 de octubre: muestra la suma de todos los movimientos.",
             "Lejos de la falla las dos órbitas no coinciden del todo (hasta ~50 cm de diferencia en 45 km). Puede ser la ionosfera, fuerte cerca del ecuador de noche (sin verificar). Lo firme es el salto junto a la falla, no los valores a decenas de km.",
             "Valores relativos dentro de la imagen; no se predicen sismos: se mide lo que ya ocurrió."],
    bounds=bounds, escala_cm=LIM_CM, capas=capas,
    camara=dict(lon=-80.75, lat=7.55, radio_m=45000, distancia_m=150000),
    epicentros=epicentros,
    falla=dict(nombre="Falla (modelo USGS, trazo aproximado)", puntos=traza,
               fuente=f"Falla: borde superior del modelo de falla finita del USGS para us6000u18k (versión {ff['properties'].get('version')}).",
               nota="Borde superior del plano del modelo de falla finita; no es el trazo exacto en superficie. Ese modelo usó también datos de radar, así que no es una comparación independiente."),
)
json.dump(P, open(os.path.join(SALIDA, "punto.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ij = os.path.join(WEB, "index.json"); I = json.load(open(ij, encoding="utf-8"))
if "panama" not in I["puntos"]:
    I["puntos"].append("panama"); json.dump(I, open(ij, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Listo:", os.path.join(SALIDA, "punto.json"))
