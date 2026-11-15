"""
agua_delta_export_visor.py -- LandSense
Lleva al visor la escena "Delta del Parana" (serie temporal GCOV, 7 fechas) como punto de tipo "serie".

Capas (una imagen por fecha, reproyectadas de UTM a grilla regular lon/lat, vecino mas cercano):
  1. "eco"    : colores falsos estilo NISAR (R = HH, G = HV, B = HH, en dB): vegetacion verde,
                superficies lisas/duras magenta, agua negra.
  2. "cambio" : cambio FUERTE de HV (dB) respecto de la media de las 3 primeras fechas (26-jun, 8-jul, 20-jul).
                Solo se pinta |cambio| >= 3 dB (umbral del control: sin evento solo 0,1-1,8% del area lo supera);
                lo demas es transparente. Azul = el eco se apago, rojo = subio. Escala de color +-6 dB.

Estado: el CAMBIO esta VERIFICADO (control: en intervalos sin evento el HV cae > 3 dB en 0,2-1,8% del
area; entre 20-jul y 13-ago en 10,7%, HH y HV juntos 8,5% vs <= 1,3%; calibracion relativa ~1; no es
fuego segun FIRMS, src/agua_delta_firms.py; reproducible). La CAUSA (agua cubriendo las islas) es HIPOTESIS.

Requiere: src/agua_gcov_exploratorio.py corrido (data/processed/gcov_exploratorio/delta_parana.npz).
Agrega "delta_parana" a web/puntos/index.json si no esta.
Uso:
    python src/agua_delta_export_visor.py
"""
import os, json
from visor_json import guardar_punto
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

AQUI = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(AQUI, "..", "data", "processed", "gcov_exploratorio", "delta_parana.npz")
WEB = os.path.join(AQUI, "..", "web", "puntos")
ID = "delta_parana"
SAL = os.path.join(WEB, ID)
os.makedirs(SAL, exist_ok=True)
PASO_GRADOS = 0.001          # ~100 m, igual que el promedio de src/agua_gcov_exploratorio.py
ESCALA_DB = 6.0
UMBRAL_DB = 3.0            # temblor normal del eco (del control): por debajo no se pinta
MESES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]
bonita = lambda f: f"{int(f[6:])} de {MESES[int(f[4:6])]} de {f[:4]}"

z = np.load(NPZ)
HH, HV = z["HH"], z["HV"]
fechas = [str(f) for f in z["fechas"]]
x0, y0, dx, dy, epsg = float(z["x0"]), float(z["y0"]), float(z["dx"]), float(z["dy"]), int(z["epsg"])
ny, nx = HV.shape[1:]
# centro de cada bloque (x0/y0 son el centro del primer pixel original de 20 m; bloque de k pixeles)
paso_orig = abs(dx) / round(abs(dx) / 20.0) if abs(dx) >= 20 else abs(dx)
cx0 = x0 + np.sign(dx) * (abs(dx) - paso_orig) / 2
cy0 = y0 + np.sign(dy) * (abs(dy) - paso_orig) / 2

# ---------- grilla lon/lat de salida ----------
a_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
esquinas_x = [cx0, cx0 + dx * (nx - 1)]; esquinas_y = [cy0, cy0 + dy * (ny - 1)]
lo, la = a_ll.transform(*np.meshgrid(esquinas_x, esquinas_y))
W, E, S, N = lo.min(), lo.max(), la.min(), la.max()
lons = np.arange(W, E, PASO_GRADOS) + PASO_GRADOS / 2
lats = np.arange(N, S, -PASO_GRADOS) - PASO_GRADOS / 2
LO, LA = np.meshgrid(lons, lats)
XX, YY = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(LO, LA)
ci = np.round((XX - cx0) / dx).astype(int); ri = np.round((YY - cy0) / dy).astype(int)
dentro = (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny)
def a_lonlat(a):
    out = np.full(LO.shape, np.nan, np.float32)
    out[dentro] = a[ri[dentro], ci[dentro]]
    return out
bounds = dict(west=float(lons[0] - PASO_GRADOS / 2), east=float(lons[-1] + PASO_GRADOS / 2),
              south=float(lats[-1] - PASO_GRADOS / 2), north=float(lats[0] + PASO_GRADOS / 2))

# ---------- capa 1: colores falsos ----------
def escala(db, lo_, hi_):
    return np.clip((db - lo_) / (hi_ - lo_), 0, 1)
arch_eco, arch_cambio = [], []
for i, f in enumerate(fechas):
    hh = a_lonlat(10 * np.log10(HH[i])); hv = a_lonlat(10 * np.log10(HV[i]))
    rgba = np.zeros(LO.shape + (4,), np.float32)
    rgba[..., 0] = escala(hh, -22, -2); rgba[..., 1] = escala(hv, -30, -10); rgba[..., 2] = escala(hh, -22, -2)
    rgba[..., 3] = np.where(np.isfinite(hh) & np.isfinite(hv), 1.0, 0.0)
    rgba[~np.isfinite(rgba)] = 0
    nombre = f"eco_{f}.png"; plt.imsave(os.path.join(SAL, nombre), rgba); arch_eco.append(nombre)

# ---------- capa 2: cambio FUERTE de HV vs referencia ----------
# Solo se pinta lo que supera el "temblor" normal del eco: |cambio| >= UMBRAL_DB. El umbral sale del control:
# en los intervalos sin evento, solo 0,1-1,8% del area cambia mas de 3 dB. Lo demas queda transparente.
ref = np.nanmean(HV[:3], 0)
cmap = plt.get_cmap("RdBu_r")
for i, f in enumerate(fechas):
    d = a_lonlat(10 * np.log10(HV[i] / ref))
    rgba = cmap(np.clip((d + ESCALA_DB) / (2 * ESCALA_DB), 0, 1)).astype(np.float32)
    fuerte = np.isfinite(d) & (np.abs(d) >= UMBRAL_DB)
    rgba[..., 3] = np.where(fuerte, 0.9, 0.0)
    nombre = f"cambio_{f}.png"; plt.imsave(os.path.join(SAL, nombre), np.nan_to_num(rgba)); arch_cambio.append(nombre)
    print(f"  {f}: {100*np.mean(d[np.isfinite(d)] <= -UMBRAL_DB):4.1f}% se apago fuerte | {100*np.mean(d[np.isfinite(d)] >= UMBRAL_DB):4.1f}% subio fuerte")

# ---------- control para la ficha ----------
i0, i1 = fechas.index("20260720"), fechas.index("20260813")
d_ev = 10 * np.log10(HV[i1] / HV[i0]); ok = np.isfinite(d_ev)
pct_ev = float(np.mean(d_ev[ok] < -3) * 100)
pct_ctrl = [float(np.mean((10 * np.log10(HV[j + 1] / HV[j]))[ok] < -3) * 100) for j in range(len(fechas) - 1) if j != i0]
print(f"Cambio evento (20-jul -> 13-ago): {pct_ev:.1f}% del area cae > 3 dB | otros intervalos: {min(pct_ctrl):.1f}-{max(pct_ctrl):.1f}%")

# ---------- grafico: % de agua segun OPERA DSWx (NASA), de src/agua_delta_dswx.py ----------
grafico = None
arch_dswx = os.path.join(os.path.dirname(NPZ), "dswx_serie.json")
if os.path.exists(arch_dswx):
    dswx = json.load(open(arch_dswx, encoding="utf-8"))
    MINV = {"HLS (optico)": 0.95, "S1 (radar C)": 0.85}     # solo fechas casi sin nubes / con cobertura completa
    ser = []
    for nombre, filas in dswx.items():
        buenas = [r for r in filas if r["validos"] >= MINV.get(nombre, 0.9)]
        etiqueta = "óptico" if "HLS" in nombre else "radar Sentinel-1"
        ser.append(dict(nombre=f"Zonas que marcó NISAR ({etiqueta})", color="#2166ac", punteada="S1" in nombre,
                        puntos=[[r["fecha"], round(r["marcado"], 1)] for r in buenas]))
        ser.append(dict(nombre=f"Zonas sin cambio, control ({etiqueta})", color="#8a8a8a", punteada="S1" in nombre,
                        puntos=[[r["fecha"], round(r["control"], 1)] for r in buenas]))
    grafico = dict(titulo="% de agua según los mapas de agua de NASA (OPERA DSWx)", eje="% con agua",
                   desde="20260612", hasta="20260925", hueco=["20260727", "20260810"], series=ser,
                   nota="Línea continua: mapa óptico (Landsat/Sentinel-2, solo días despejados). Punteada: mapa de radar "
                        "Sentinel-1 (cuenta solo agua abierta, por eso da menos). Franja: días sin datos de NISAR.")
    print(f"Grafico DSWx: {sum(len(x['puntos']) for x in ser)} puntos")
else:
    print("AVISO: falta dswx_serie.json (correr src/agua_delta_dswx.py): la escena sale sin grafico ni validacion de agua")

punto = dict(
    id=ID, tipo="serie",
    nombre="Delta del Paraná: el humedal visto por radar",
    lugar="Delta inferior del río Paraná, Buenos Aires y Entre Ríos, Argentina",
    titular=("A comienzos de agosto el agua cubrió los bajos de las islas del Delta; NISAR lo vio a través de las nubes"
             if grafico else f"En agosto, el radar dejó de recibir eco en cerca del {pct_ev:.0f}% del Delta"),
    que_es=("Esto no es una foto: son 7 pasadas del radar NISAR entre junio y septiembre de 2026, una cada 12 días. "
            "El radar mide cuánto eco devuelve cada lugar: la vegetación devuelve mucho, el agua quieta casi nada. "
            "Mové la línea de tiempo para ver cómo cambió el humedal."),
    estado="VALIDADO" if grafico else "VERIFICADO",
    estado_detalle=((f"Validado con dos mapas de agua de NASA (OPERA DSWx), independientes de NISAR: en las zonas donde el "
                     "radar vio apagarse el eco, el agua pasó de alrededor del 5% en junio y julio a cerca del 70% a fines de "
                     "agosto (mapa óptico) y bajó en septiembre; en las zonas sin cambio casi no se movió. El anegamiento "
                     "empezó entre el 28 de julio y el 7 de agosto. Lo que NO sabemos es por qué: desborde del río, lluvia "
                     "o sudestada.") if grafico else
                    (f"El cambio está verificado: entre el 20 de julio y el 13 de agosto el eco cayó más de 3 dB en el "
                     f"{pct_ev:.1f}% del área, contra {min(pct_ctrl):.1f}-{max(pct_ctrl):.1f}% en los otros intervalos. "
                     "La causa no está confirmada.")),
    resultado=(f"NISAR: el eco se apagó fuerte (más de 3 dB) en el {pct_ev:.1f}% del área entre el 20 de julio y el 13 de agosto, "
               f"contra {min(pct_ctrl):.1f}-{max(pct_ctrl):.1f}% en los intervalos sin evento. "
               "Zonas que cambiaron: el eco cae en las dos polarizaciones a la vez (HH y HV, ~5 dB), parte pasa a verse como "
               "agua abierta, y se recupera de a poco (−4,8 dB el 25-ago, −3,6 el 6-sep, −2,7 el 18-sep). "
               "Sobre 417 focos FIRMS en la zona, el eco donde hubo fuego casi no cambió: la banda L ve bien el fuego en "
               "bosque, no en pastizal."),
    evento="Cambio del humedal entre el 20 de julio y el 13 de agosto de 2026 (causa sin confirmar)",
    satelite=("NISAR (NASA-ISRO), banda L, producto GCOV provisional (HH y HV), promedio ~100 m. Validación: OPERA DSWx-HLS y "
              "DSWx-S1 (NASA JPL / PO.DAAC), mapas de agua de 30 m"),
    pares="Órbita 010 descendente, 7 fechas: " + ", ".join(bonita(f) for f in fechas),
    limites=[
        "Que es agua está confirmado; por qué llegó el agua (río, lluvia o viento) no.",
        "Los porcentajes de agua de NASA varían hasta ~15 puntos entre días y sensores: sirven para ver la tendencia, no como cifra exacta de un día.",
        "Entre el 27 de julio y el 10 de agosto NISAR no tomó datos (hueco del instrumento); los mapas de agua de NASA ubican el comienzo del anegamiento entre el 28 de julio y el 7 de agosto.",
        "El intervalo del cambio dura 24 días; los demás, 12.",
        "Promedio a unos 100 m: no se ven detalles más chicos.",
        "El radar no ve bien incendios de pastizal ni junco (sí de bosque).",
    ],
    bounds=bounds,
    fechas=fechas,
    fechas_texto=[bonita(f) for f in fechas],
    fecha_referencia="la media del 26 de junio, 8 de julio y 20 de julio",
    capas=[
        dict(id="eco", nombre="Qué ve el radar", archivos=arch_eco,
             leyenda=dict(titulo="Eco del radar (colores falsos, estilo NISAR)",
                          gradiente="linear-gradient(90deg,#000,#1a8f1a,#d23cd2)",
                          izq="poco eco", der="mucho eco",
                          palabras="<b>Negro</b>: agua quieta (casi sin eco).<br><b>Verde</b>: vegetación.<br>"
                                   "<b>Magenta</b>: superficies duras o lisas (suelo desnudo, construcciones).")),
        dict(id="cambio", nombre="Qué cambió", archivos=arch_cambio,
             leyenda=dict(titulo="Dónde cambió fuerte el eco (respecto de fines de junio y julio)",
                          gradiente="linear-gradient(90deg,#2166ac 0%,#92c5de 45%,transparent 45%,transparent 55%,#f4a582 55%,#b2182b 100%)",
                          izq=f"se apagó (−{ESCALA_DB:.0f} dB)", der=f"subió (+{ESCALA_DB:.0f} dB)",
                          palabras="<b style='color:#2166ac'>Azul</b>: el eco se apagó fuerte (por ejemplo, agua que cubre la vegetación).<br>"
                                   "<b style='color:#b2182b'>Rojo</b>: el eco subió fuerte.<br>"
                                   f"Sin color: cambió menos de {UMBRAL_DB:.0f} dB, el temblor normal del radar (sin evento, "
                                   "solo 0,1-1,8% del Delta lo supera).<br>"
                                   "Los cuadraditos del sudoeste son campos: ahí los cambios son trabajo agrícola, no el humedal.")),
    ],
    inicio=dict(capa="cambio", fecha="20260813"),
    camara=dict(lon=float((W + E) / 2), lat=float((S + N) / 2), radio_m=25000, distancia_m=80000),
    marcadores=[],
    grafico=grafico,
)
guardar_punto(os.path.join(SAL, "punto.json"), punto)

idx_path = os.path.join(WEB, "index.json")
idx = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else {"puntos": []}
if ID not in idx["puntos"]:
    idx["puntos"].append(ID)
    json.dump(idx, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Exportado: {SAL} ({len(fechas)} fechas x 2 capas) | index.json: {idx['puntos']}")
