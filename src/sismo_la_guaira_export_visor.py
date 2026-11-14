"""
sismo_la_guaira_export_visor.py -- LandSense
Exporta el desplazamiento co-sismico VERIFICADO de La Guaira (src/sismo_la_guaira_descomposicion.py) al visor 3D.

Por que un script propio:
  El visor viejo estiraba una grilla UTM (metros) sobre un rectangulo lon/lat.
  Cesium SingleTileImageryProvider usa un esquema GEOGRAFICO: la imagen tiene que
  estar en una grilla REGULAR de lon/lat, si no queda rotada/corrida sobre el terreno
  (en 100 km de ancho, cientos de metros). Aca se REPROYECTA de verdad:
  para cada pixel de salida (lon, lat) se busca el pixel UTM mas cercano.

Salidas en ../web/puntos/la_guaira/:
  este_oeste.png, vertical.png  (RGBA, transparente donde no hay dato)
  punto.json                    (bounds, escala, textos, fuente, estado, limites,
                                 epicentros y falla del USGS, flechas de movimiento)

Colores como NASA: rojo = este / sube; azul = oeste / baja. Escala fija simetrica.
Probado con el dato real (percentiles identicos antes/despues de reproyectar).
Requiere internet (baja epicentros y modelo de falla del USGS en vivo).
Uso:
    python src/sismo_la_guaira_export_visor.py
"""
import os, json
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

AQUI = os.path.dirname(os.path.abspath(__file__))
ENTRADA = os.path.join(AQUI, "..", "data", "processed", "la_guaira", "EU_coseismico.npz")
SALIDA = os.path.join(AQUI, "..", "web", "puntos", "la_guaira")
os.makedirs(SALIDA, exist_ok=True)

LIM_CM = 60.0          # escala fija +-60 cm (NASA reporta hasta ~60 cm)
PASO_GRADOS = 0.0006   # ~66 m por pixel de salida (el dato es de 80 m)

assert os.path.exists(ENTRADA), "falta EU_coseismico.npz: correr src/sismo_la_guaira_descomposicion.py primero"
z = np.load(ENTRADA)
E, U, x, y, epsg = z["E_cm"], z["U_cm"], z["x"], z["y"], int(z["epsg"])
assert E.shape == (len(y), len(x)) and U.shape == E.shape, "forma inesperada"

# Extension real del dato valido, en grados
filas = np.where(np.isfinite(E).any(1))[0]; cols = np.where(np.isfinite(E).any(0))[0]
X, Y = np.meshgrid(x[cols.min():cols.max() + 1], y[filas.min():filas.max() + 1])
a_geo = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
lon, lat = a_geo.transform(X, Y)
oeste, este = float(np.nanmin(lon)), float(np.nanmax(lon))
sur, norte = float(np.nanmin(lat)), float(np.nanmax(lat))

# Grilla regular lon/lat de salida (fila 0 = norte, como una imagen)
lons = np.arange(oeste, este, PASO_GRADOS) + PASO_GRADOS / 2
lats = np.arange(norte, sur, -PASO_GRADOS) - PASO_GRADOS / 2
LON, LAT = np.meshgrid(lons, lats)
a_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
XU, YU = a_utm.transform(LON, LAT)
dx, dy = x[1] - x[0], y[1] - y[0]
ix = np.round((XU - x[0]) / dx).astype(int); iy = np.round((YU - y[0]) / dy).astype(int)
ok = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))

def reproyectar(campo):
    out = np.full(LON.shape, np.nan, np.float32)
    out[ok] = campo[iy[ok], ix[ok]]
    return out

# Bordes exactos de la imagen (lo que Cesium tiene que recibir)
bounds = dict(west=float(lons[0] - PASO_GRADOS / 2), east=float(lons[-1] + PASO_GRADOS / 2),
              north=float(lats[0] + PASO_GRADOS / 2), south=float(lats[-1] - PASO_GRADOS / 2))

def guardar(campo, nombre):
    rgba = plt.get_cmap("RdBu_r")(np.clip((campo + LIM_CM) / (2 * LIM_CM), 0, 1))
    rgba[..., 3] = np.where(np.isfinite(campo), 1.0, 0.0)
    plt.imsave(os.path.join(SALIDA, nombre), rgba)
    v = campo[np.isfinite(campo)]
    return dict(archivo=nombre, validos=int(v.size), p1=round(float(np.percentile(v, 1)), 1),
                p99=round(float(np.percentile(v, 99)), 1))

Er, Ur = reproyectar(E), reproyectar(U)
# Chequeo de la reproyeccion: los estadisticos no deben cambiar (mismo dato, otra grilla)
for nom, orig, rep in [("E", E, Er), ("U", U, Ur)]:
    po, pr = np.nanpercentile(orig, [1, 50, 99]), np.nanpercentile(rep, [1, 50, 99])
    print(f"{nom}: original p1/p50/p99 {po.round(1)} | reproyectado {pr.round(1)}")
    assert np.all(np.abs(po - pr) < 3.0), f"la reproyeccion de {nom} cambio el dato: revisar"

capaE = guardar(Er, "este_oeste.png")
capaU = guardar(Ur, "vertical.png")


# ---------- Datos oficiales del USGS (se bajan en vivo: reproducible y con fuente) ----------
USGS_EV = "https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={}&format=geojson"
epicentros = []
for ev in ["us6000t7zp", "us6000t7zc"]:
    d = requests.get(USGS_EV.format(ev), timeout=60).json()
    lo, la, prof = d["geometry"]["coordinates"]
    epicentros.append(dict(id=ev, mag=d["properties"]["mag"], lugar=d["properties"]["place"],
                           lon=lo, lat=la, prof_km=prof,
                           dentro=bool(bounds["west"] <= lo <= bounds["east"] and bounds["south"] <= la <= bounds["north"])))
    print(f"USGS {ev}: M{d['properties']['mag']} {d['properties']['place']} ({lo:.3f}, {la:.3f}) dentro del mapa: {epicentros[-1]['dentro']}")

# Traza aproximada: borde superior del modelo de falla finita del USGS (plano simplificado)
det = requests.get(USGS_EV.format("us6000t7zp"), timeout=60).json()
ffm_url = det["properties"]["products"]["finite-fault"][0]["contents"]["FFM.geojson"]["url"]
ffm = requests.get(ffm_url, timeout=60).json()
sup = []
for f in ffm["features"]:
    c = np.array(f["geometry"]["coordinates"][0])
    sup += [tuple(p) for p in c if p[2] <= c[:, 2].min() + 1]
sup = np.array(sup); sup = sup[sup[:, 2] <= sup[:, 2].min() + 50]
traza = sorted({(round(float(p[0]), 3), round(float(p[1]), 3)) for p in sup})
print(f"Traza del modelo USGS: {len(traza)} puntos, lon {traza[0][0]}..{traza[-1][0]}")

# ---------- Flechas de movimiento (solo componente este-oeste medida) ----------
PASO_FLECHA = 0.03       # grados (~3.3 km) entre flechas
# Umbral: el mapa es RELATIVO a la zona de referencia (lat < 10.40, src/sismo_la_guaira_descomposicion.py). El "cero" es
# ambiguo en lo que varia E DENTRO de esa zona (medido: p5 -12.8 / p95 +9.8 cm).
# Solo se dibujan flechas que superan esa ambiguedad y 2x el ruido de control (~4 cm).
ref = np.isfinite(Er) & (LAT < 10.40)
p5_ref, p95_ref = np.percentile(Er[ref], [5, 95])
MIN_CM = float(max(8.0, abs(p5_ref), abs(p95_ref)))
print(f"Zona de referencia: E p5 {p5_ref:.1f} / p95 {p95_ref:.1f} cm -> umbral de flechas {MIN_CM:.1f} cm")
EXAGERA_M_POR_CM = 50.0  # 1 cm de movimiento = 50 m de flecha (exageracion x5000)
flechas = []
for lo in np.arange(bounds["west"] + PASO_FLECHA / 2, bounds["east"], PASO_FLECHA):
    for la in np.arange(bounds["south"] + PASO_FLECHA / 2, bounds["north"], PASO_FLECHA):
        m = (np.abs(LON - lo) <= PASO_FLECHA / 2) & (np.abs(LAT - la) <= PASO_FLECHA / 2)
        v = Er[m]; v = v[np.isfinite(v)]
        if m.sum() == 0 or v.size < 0.3 * m.sum():
            continue                                  # poca cobertura: sin flecha
        e = float(np.median(v))
        if abs(e) < MIN_CM:
            continue
        dlon = e * EXAGERA_M_POR_CM / (111320 * np.cos(np.radians(la)))
        flechas.append(dict(lon=round(float(lo), 4), lat=round(float(la), 4),
                            lon2=round(float(lo + dlon), 4), cm=round(e, 1)))
print(f"Flechas: {len(flechas)} (|E| >= {MIN_CM} cm)")

punto = {
    "id": "la_guaira",
    "nombre": "Terremoto de La Guaira",
    "lugar": "La Guaira y Caracas, Venezuela",
    "tipo": "antes_despues",
    "titular": "Durante el terremoto, el suelo se corrió hasta unos 56 cm hacia el oeste",
    "que_es": ("Este mapa no es una foto: compara dos pasadas del radar NISAR, una antes y otra después "
               "del sismo, y muestra cuántos centímetros se movió cada punto del suelo entre esas fechas."),
    "fecha_antes": "18 de junio de 2026",
    "fecha_despues": "30 de junio de 2026",
    "evento": "Sismos M7.2 y M7.5 del 24 de junio de 2026 (falla de rumbo, sistema San Sebastián)",
    "satelite": "NISAR (NASA-ISRO), banda L, productos GUNW provisionales",
    "pares": "Ascendente 18 jun - 30 jun (track 061) y descendente 22 jun - 4 jul (track 126)",
    "estado": "VERIFICADO",
    "estado_detalle": ("Señal solo en los pares que cruzan el sismo (4 órbitas, cada una con su control); "
                       "dos pares ascendentes independientes coinciden (r = 0,98)."),
    "resultado": ("Unos 56 cm hacia el oeste al sur del aeropuerto de Maiquetía; movimiento vertical "
                  "cercano a 2 cm. NASA/JPL reportó hasta unos 60 cm hacia el oeste en la misma zona."),
    "limites": [
        "Valores relativos a una zona de referencia 20 km al sur; esa zona varía unos 13 cm, "
        "así que movimientos menores a eso no se interpretan.",
        "Supone movimiento norte-sur nulo.",
        "Ruido medido en los pares de control: unos 4 cm.",
        "No se predicen sismos: se mide lo que ya ocurrió."
    ],
    "bounds": bounds,
    "escala_cm": LIM_CM,
    "capas": [
        dict(id="este_oeste", nombre="Este-oeste", izquierda="oeste", derecha="este",
             texto_izq="el suelo se movió hacia el oeste", texto_der="el suelo se movió hacia el este",
             texto_neutro="blanco: casi no se movió", **capaE),
        dict(id="vertical", nombre="Vertical", izquierda="baja", derecha="sube",
             texto_izq="el suelo bajó", texto_der="el suelo subió",
             texto_neutro="blanco: casi no subió ni bajó", **capaU),
    ],
    "camara": dict(lon=-67.05, lat=10.52, radio_m=30000, distancia_m=90000),
    "marcadores": [dict(nombre="Aeropuerto de Maiquetía", lon=-66.99, lat=10.60)],
    "epicentros": epicentros,
    "falla": dict(nombre="Falla (modelo USGS, trazo aproximado)", puntos=traza,
                  fuente=f"USGS finite fault us6000t7zp: {ffm_url}",
                  nota="Borde superior del plano del modelo de falla finita; no es el trazo exacto en superficie."),
    "flechas": dict(datos=flechas, exageracion=int(EXAGERA_M_POR_CM * 100), minimo_cm=MIN_CM,
                    nota=("Solo dirección este-oeste (se supone norte-sur nulo). Largo exagerado: "
                          "1 cm de movimiento = 50 m de flecha. Solo se dibujan movimientos mayores a "
                          f"{MIN_CM:.0f} cm, que es cuánto varía la zona de referencia.")),
}
json.dump(punto, open(os.path.join(SALIDA, "punto.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("bounds:", bounds)
print("Listo:", os.path.abspath(SALIDA))
