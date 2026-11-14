"""
radar_ojos_nisar_export.py -- LandSense
"LOS OJOS DE NISAR": exporta para el visor (Cesium) las imagenes reales del radar NISAR (GCOV, banda L)
fecha por fecha, reproyectadas a una grilla regular lon/lat para que se apoyen sobre el terreno.
Es SOLO IMAGEN (no afirma nada): la misma composicion de src/radar_imagenes.py.
  rojo = HH (-20..-3 dB) | verde = HV (-25..-10 dB) | azul = HH/HV (0..15 dB) x brillo
  NEGRO = agua quieta. Verde-amarillo = vegetacion. Rosado/blanco = ciudad.

Zonas:
  libres    -> escena NUEVA web/puntos/libres (serie de pasadas + zonas que src/agua_crecida_uruguay.py marco el 12-oct)
  apipe     -> escena NUEVA web/puntos/apipe  (serie de pasadas, sin analisis: estado "Solo imagen")
  la_guaira -> agrega la capa "ojos" (antes / despues) a la escena existente web/puntos/la_guaira

Uso:
    python src/radar_ojos_nisar_export.py libres apipe la_guaira
    python src/radar_ojos_nisar_export.py --prueba     (sintetico, sin internet)
Requiere para 'libres': data/processed/crecida/uruguay/libres_010D_103.npz (src/agua_crecida_uruguay.py).
Despues: Ctrl+F5 en el visor (Live Server).
"""
import os, re, sys, json, time
from collections import defaultdict
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(AQUI, "..", "web", "puntos")
GCOV = "C2854338529-ASF"
G = "science/LSAR/GCOV/grids/frequencyA"
ESCALA = dict(R=(-20.0, -3.0), G=(-25.0, -10.0), B=(0.0, 15.0))   # igual que src/radar_imagenes.py
PASO_GRADOS = 0.0003            # ~30 m: liviano para la web y nitido sobre el terreno
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

ZONAS = {
    "libres": dict(bbox=(-57.25, -29.85, -56.95, -29.55), orbita=("010", "D", "103")),
    "apipe":  dict(bbox=(-56.95, -27.60, -56.55, -27.40), orbita=("010", "D", "102")),
    "chaco":  dict(bbox=(-59.05, -27.60, -58.55, -27.00), orbita=("111", "D", "102")),   # crecida Gran Resistencia (src/agua_crecida_chaco.py, VALIDADO)
    "la_guaira": dict(bbox=None, orbita=("126", "D", None), antes="20260610", despues="20260704"),
}

# ---------------- logica pura ----------------
def suavizar(a, k=3):
    v = np.isfinite(a); x = np.where(v, a, 0.0); r = k // 2
    xp = np.pad(x, r); vp = np.pad(v.astype(float), r)
    S = sum(xp[i:i + a.shape[0], j:j + a.shape[1]] for i in range(k) for j in range(k))
    N = sum(vp[i:i + a.shape[0], j:j + a.shape[1]] for i in range(k) for j in range(k))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = S / N
    out[~v] = np.nan
    return out

def rgba(hh, hv):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 10 * np.log10(hh); g = 10 * np.log10(hv); b = r - g
    esc = lambda x, lo, hi: np.clip((x - lo) / (hi - lo), 0, 1)
    R, Gc = esc(r, *ESCALA["R"]), esc(g, *ESCALA["G"])
    B = esc(b, *ESCALA["B"]) * np.maximum(R, Gc)
    A = (np.isfinite(r) & np.isfinite(g)).astype(float)
    img = np.dstack([R, Gc, B, A]); img[A == 0] = 0
    return img

def grilla_lonlat(bounds, paso=PASO_GRADOS):
    w, s, e, n = bounds["west"], bounds["south"], bounds["east"], bounds["north"]
    nx = int(round((e - w) / paso)); ny = int(round((n - s) / paso))
    lon = w + (np.arange(nx) + 0.5) * (e - w) / nx
    lat = n - (np.arange(ny) + 0.5) * (n - s) / ny          # fila 0 = norte (como espera Cesium)
    return np.meshgrid(lon, lat)

def remuestrear(img, X, Y, x0, y0, dx, dy):
    """Vecino mas cercano: img (ny,nx,...) con centro del pixel (0,0) en (x0,y0) y paso (dx,dy)."""
    j = np.rint((X - x0) / dx).astype(int); i = np.rint((Y - y0) / dy).astype(int)
    ok = (i >= 0) & (i < img.shape[0]) & (j >= 0) & (j < img.shape[1])
    out = np.zeros(X.shape + img.shape[2:], img.dtype)
    out[ok] = img[i[ok], j[ok]]
    return out

def fecha_texto(f):
    return f"{int(f[6:])} de {MESES[int(f[4:6]) - 1]} de {f[:4]}"

def prueba():
    # 1) composicion: agua negra/transparente no; agua = negro opaco; sin dato = transparente
    hh = np.full((10, 10), 10 ** (-28 / 10)); hv = np.full((10, 10), 10 ** (-33 / 10))
    hh[:, 5:] = 10 ** (-8 / 10); hv[:, 5:] = 10 ** (-12 / 10); hh[0, 0] = np.nan
    im = rgba(hh, hv)
    assert im[5, 2, :3].max() < 0.05 and im[5, 2, 3] == 1, "agua: negro opaco"
    assert im[5, 7, 1] > 0.8, "monte: verde"
    assert im[0, 0, 3] == 0, "sin dato: transparente"
    # 2) remuestreo: una grilla UTM sintetica con una marca en un pixel conocido
    img = np.zeros((50, 40, 4)); img[10, 20] = 1
    x0, y0, dx, dy = 1000.0, 5000.0, 20.0, -20.0
    X = np.array([[x0 + 20 * dx]]); Y = np.array([[y0 + 10 * dy]])
    assert remuestrear(img, X, Y, x0, y0, dx, dy)[0, 0, 3] == 1, "remuestreo corrido"
    X2 = np.array([[x0 - 500]]); assert remuestrear(img, X2, Y, x0, y0, dx, dy)[0, 0, 3] == 0, "fuera = transparente"
    # 3) grilla lon/lat: fila 0 al norte
    LO, LA = grilla_lonlat(dict(west=-57.0, east=-56.9, south=-29.8, north=-29.7), 0.01)
    assert LA[0, 0] > LA[-1, 0] and LO[0, 0] < LO[0, -1] and LO.shape == (10, 10)
    assert fecha_texto("20261012") == "12 de octubre de 2026"
    print("[prueba] OK: agua negra opaca, monte verde, sin dato transparente; remuestreo y grilla correctos.")

# ---------------- datos reales ----------------
def buscar(bbox, orbita):
    import requests
    w, s, e, n = bbox
    r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", timeout=60, params=dict(
        collection_concept_id=GCOV, bounding_box=f"{w},{s},{e},{n}",
        temporal="2026-06-17T00:00:00Z,2026-12-31T00:00:00Z", page_size=500)).json()["feed"]["entry"]
    g = defaultdict(dict)
    for ent in r:
        m = re.search(r"GCOV_\d{3}_(\d{3})_([AD])_(\d{3})_\d{4}_(\w{4})_[A-Z]_(\d{8})T.*_(\d{3})$", ent["title"])
        h5 = [l["href"] for l in ent.get("links", []) if l["href"].startswith("https") and l["href"].endswith(".h5")]
        if not (m and h5 and m.group(4) == "DHDH"): continue
        if m.group(1) != orbita[0] or m.group(2) != orbita[1] or (orbita[2] and m.group(3) != orbita[2]): continue
        k = (m.group(1), m.group(2), m.group(3)); f = m.group(5)
        if f not in g[k] or m.group(6) > g[k][f][0]: g[k][f] = (m.group(6), h5[0])
    if not g: return None, {}
    k = max(g, key=lambda k: len(g[k]))
    return k, {f: u for f, (v, u) in g[k].items()}

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
                c = np.where((x >= min(xs) - 200) & (x <= max(xs) + 200))[0]
                r = np.where((y >= min(ys) - 200) & (y <= max(ys) + 200))[0]
                if c.size < 20 or r.size < 20: return None
                hh = f[f"{G}/HHHH"][r.min():r.max() + 1, c.min():c.max() + 1].astype(np.float32)
                hv = f[f"{G}/HVHV"][r.min():r.max() + 1, c.min():c.max() + 1].astype(np.float32)
            bad = ~(np.isfinite(hh) & np.isfinite(hv) & (hh > 0) & (hv > 0)); hh[bad] = np.nan; hv[bad] = np.nan
            return dict(hh=hh, hv=hv, x0=float(x[c.min()]), y0=float(y[r.min()]),
                        dx=float(x[1] - x[0]), dy=float(y[1] - y[0]), epsg=epsg)
        except Exception as ex:
            print(f"    reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})", flush=True); time.sleep(6)
    return None

def guardar_png(img, path):
    from PIL import Image
    a = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    metodo = getattr(getattr(Image, "Quantize", None), "FASTOCTREE", 2)   # Pillow viejo: 2 = FASTOCTREE
    Image.fromarray(a, "RGBA").quantize(colors=200, method=metodo).save(path, optimize=True)

def promediar(a, k):
    """Promedio en bloques k x k (potencia lineal), ignorando NaN. Baja memoria en zonas grandes."""
    if k <= 1: return a
    ny, nx = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    with np.errstate(invalid="ignore"):
        return np.nanmean(a[:ny, :nx].reshape(ny // k, k, nx // k, k), axis=(1, 3)).astype(np.float32)

def imagen_lonlat(d, bounds, objetivo_m=30.0):
    from pyproj import Transformer
    k = max(1, int(round(objetivo_m / abs(d["dx"]))))
    hh, hv = promediar(d["hh"], k), promediar(d["hv"], k)
    x0 = d["x0"] + (k - 1) / 2 * d["dx"]; y0 = d["y0"] + (k - 1) / 2 * d["dy"]     # centro del bloque (0,0)
    k_suave = 3 if k == 1 else 1                       # si ya se promedio, no hace falta suavizar
    if k_suave > 1: hh, hv = suavizar(hh), suavizar(hv)
    LO, LA = grilla_lonlat(bounds)
    X, Y = Transformer.from_crs("EPSG:4326", f"EPSG:{d['epsg']}", always_xy=True).transform(LO, LA)
    return remuestrear(rgba(hh, hv).astype(np.float32), X, Y, x0, y0, d["dx"] * k, d["dy"] * k)

def bounds_de(bbox):
    return dict(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3])

LEYENDA_OJOS = {"titulo": "Los ojos de NISAR: el eco del radar (colores falsos)",
                "gradiente": "linear-gradient(90deg,#000 0%,#000 18%,#6b7d1e 40%,#c8e06a 62%,#e9b7d6 85%,#fff 100%)",
                "izq": "sin eco (agua)", "der": "mucho eco",
                "palabras": "<b>Negro</b>: agua quieta, que hace de espejo.<br><b>Verde-amarillo</b>: vegetación.<br><b>Rosado o blanco</b>: ciudad y construcciones.<br>Es la imagen real del radar, no una foto: no tiene nubes."}

def export_serie(zona, cfg):
    k, fechas = buscar(cfg["bbox"], cfg["orbita"])
    if not fechas: print(f"  {zona}: sin GCOV"); return None
    print(f"  {zona}: orbita {k} | {len(fechas)} fechas: {sorted(fechas)}", flush=True)
    od = os.path.join(WEB, zona); os.makedirs(od, exist_ok=True)
    b = bounds_de(cfg["bbox"]); hechas = []
    for f in sorted(fechas):
        d = leer(fechas[f], cfg["bbox"])
        if d is None: print(f"   {f}: sin datos en la zona"); continue
        img = imagen_lonlat(d, b)
        guardar_png(img, os.path.join(od, f"ojos_{f}.png"))
        hechas.append(f); print(f"   {f}: ojos_{f}.png ({img.shape[1]}x{img.shape[0]}, con dato {img[..., 3].mean():.0%})", flush=True)
    return k, hechas, b, od

def marcado_libres(fechas, b, od):
    from pyproj import Transformer
    npz = os.path.join(AQUI, "..", "data", "processed", "crecida", "uruguay", "libres_010D_103.npz")
    z = np.load(npz)
    M = z["marcado"]; dx, dy = float(z["dx"]), float(z["dy"]); epsg = int(z["epsg"])
    po = 20.0
    cx0 = float(z["x0"]) - np.sign(dx) * po / 2 + dx / 2; cy0 = float(z["y0"]) - np.sign(dy) * po / 2 + dy / 2   # centro de la celda (0,0)
    LO, LA = grilla_lonlat(b)
    X, Y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(LO, LA)
    m = remuestrear(M.astype(np.uint8)[..., None], X, Y, cx0, cy0, dx, dy)[..., 0] > 0
    img = np.zeros(m.shape + (4,)); img[m] = [0.25, 0.66, 1.0, 0.75]
    guardar_png(img, os.path.join(od, "marcado_20261012.png"))
    vacia = np.zeros((4, 4, 4)); guardar_png(vacia, os.path.join(od, "vacio.png"))
    print(f"   zonas marcadas: {int(M.sum())} celdas de 100 m -> marcado_20261012.png", flush=True)
    return ["marcado_20261012.png" if f == "20261012" else "vacio.png" for f in fechas]

def marcado_npz(npz, fecha_ev, fechas, b, od):
    """Igual que marcado_libres pero generico (npz de src/agua_crecida_uruguay.py o src/agua_crecida_chaco.py)."""
    from pyproj import Transformer
    z = np.load(npz)
    M = z["marcado"]; dx, dy = float(z["dx"]), float(z["dy"]); epsg = int(z["epsg"])
    po = 20.0
    cx0 = float(z["x0"]) - np.sign(dx) * po / 2 + dx / 2; cy0 = float(z["y0"]) - np.sign(dy) * po / 2 + dy / 2
    LO, LA = grilla_lonlat(b)
    X, Y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(LO, LA)
    m = remuestrear(M.astype(np.uint8)[..., None], X, Y, cx0, cy0, dx, dy)[..., 0] > 0
    img = np.zeros(m.shape + (4,)); img[m] = [0.25, 0.66, 1.0, 0.75]
    nombre = f"marcado_{fecha_ev}.png"
    guardar_png(img, os.path.join(od, nombre))
    guardar_png(np.zeros((4, 4, 4)), os.path.join(od, "vacio.png"))
    print(f"   zonas marcadas: {int(M.sum())} celdas de 100 m -> {nombre}", flush=True)
    return [nombre if f == fecha_ev else "vacio.png" for f in fechas]

def hacer_chaco():
    r = export_serie("chaco", ZONAS["chaco"])
    if not r: return
    k, fechas, b, od = r
    npz = os.path.join(AQUI, "..", "data", "processed", "crecida", "chaco", "gran_resistencia_111D_102.npz")
    marc = marcado_npz(npz, "20261019", fechas, b, od)
    extra = dict(
        nombre="Crecida del Paraná y el Paraguay en Chaco", lugar="Gran Resistencia: Isla del Cerrito, Barranqueras y Puerto Vilelas (Chaco, Argentina)",
        estado="VALIDADO", color="verde",
        titular="Durante El Niño, las lluvias hicieron crecer el Paraná y el Paraguay, y NISAR vio el agua entrar en las islas aunque estuviera nublado",
        que_es="Estas son las pasadas reales del radar NISAR entre junio y octubre de 2026, una cada 12 días. Mové la línea de tiempo: hasta el 7 de octubre las islas se ven verdes; el 19 de octubre, con el río a unos 7 metros en Barranqueras, los bajos de las islas se llenan de negro (agua).",
        estado_detalle="Hicimos la prueba a ciegas: fijamos la regla antes de mirar los datos. Dos pasadas distintas de NISAR (16 y 19 de octubre, desde dos órbitas) marcaron agua nueva, y el mapa de agua de NASA hecho con fotos de otros satélites lo confirma: en las zonas marcadas el agua pasó de 13 % (agosto) a 87 % (19 de octubre); en las zonas sin cambio, de 7 % a 8 %. Además, las dos pasadas coinciden: el 97 % de lo que NISAR marcó el 16 de octubre lo volvió a marcar el 19.",
        resultado="src/agua_crecida_chaco.py (= receta de src/agua_crecida_uruguay.py): MARCADO = caída de HV ≥ 3 dB respecto de la referencia, celdas de 100 m. 111 D (ref 20-ago + 1-sep, evento 19-oct): DSWx-HLS 13,0 % → 87,4 % en marcado, 6,6 % → 7,9 % en control, control temporal 496/16.181. 060 A (ref 17-ago + 29-ago, evento 16-oct): 26,4 % → 79,5 % y 10,0 % → 12,0 %, control temporal 752/7.898.",
        evento="Crecida de los ríos Paraná y Paraguay, octubre de 2026, durante El Niño (lluvias intensas en la cuenca). Paraná en Barranqueras: 5,40 m el 14-oct y 7,16 m el 20-oct (evacuación desde 6,50 m, Prefectura). Chaco informó 913 personas trasladadas y 281 autoevacuadas (21-oct).",
        limites=["Decimos 'durante El Niño', no 'por El Niño': NISAR ve el agua, no la causa.",
                 "NISAR mide en celdas de 100 m: ve islas y bañados inundados, no una calle o una casa.",
                 "La confirmación sale de fotos satelitales (DSWx-HLS) con una sola fecha despejada antes de la crecida (23 de agosto); el mapa de NASA hecho con otro radar (Sentinel-1) no alcanzó a cubrir la zona en las fechas del pico.",
                 "El agua debajo del monte o de pastizales altos puede no verse.",
                 "Algunas manchas celestes al norte (cerca de Las Palmas) caen sobre arroceras con riego: ahí el agua es del cultivo, no de la crecida."],
        capas=[dict(id="marcado", nombre="Agua nueva (19-oct)", archivos=marc, sobre="ojos",
                    texto_fecha="{f}: en celeste, las zonas donde NISAR vio el eco apagarse fuerte respecto de fines de agosto (solo se calcula para el 19 de octubre).",
                    leyenda={"titulo": "Agua nueva según NISAR (19 de octubre)", "gradiente": "linear-gradient(90deg,#40a8ff,#40a8ff)",
                             "izq": "", "der": "", "palabras": "<b style='color:#40a8ff'>Celeste</b>: el eco del radar se apagó fuerte (3 dB o más) respecto de fines de agosto. Las fotos satelitales de NASA confirman agua en el 87 % de estas zonas."})])
    escribir_punto("chaco", k, fechas, b, od, extra)

def escribir_punto(zona, k, fechas, b, od, extra):
    p = dict(id=zona, tipo="serie", bounds=b, fechas=fechas, fechas_texto=[fecha_texto(f) for f in fechas],
             fecha_referencia=fecha_texto(fechas[0]),
             satelite="NISAR (NASA-ISRO), radar banda L, producto GCOV PROVISIONAL (HH+HV, 20 m)",
             pares=f"Órbita {k[0]} {'descendente' if k[1] == 'D' else 'ascendente'}, frame {k[2]}: {len(fechas)} pasadas",
             camara=dict(lon=(b["west"] + b["east"]) / 2, lat=(b["south"] + b["north"]) / 2, radio_m=15000, distancia_m=48000),
             inicio=dict(capa="ojos", fecha=fechas[-1]))
    p.update(extra)
    p["capas"] = [dict(id="ojos", nombre="Ojos de NISAR", archivos=[f"ojos_{f}.png" for f in fechas], leyenda=LEYENDA_OJOS,
                       texto_fecha="{f}: así vio el radar este lugar ese día. Negro = agua.")] + p.get("capas", [])
    json.dump(p, open(os.path.join(od, "punto.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    idx_p = os.path.join(WEB, "index.json"); idx = json.load(open(idx_p, encoding="utf-8"))
    if zona not in idx["puntos"]:
        idx["puntos"].append(zona); json.dump(idx, open(idx_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  {od}/punto.json escrito", flush=True)

def hacer_libres():
    r = export_serie("libres", ZONAS["libres"])
    if not r: return
    k, fechas, b, od = r
    marc = marcado_libres(fechas, b, od)
    extra = dict(
        nombre="Crecida del río Uruguay", lugar="Paso de los Libres (Argentina) y Uruguaiana (Brasil)",
        estado="Indicio fuerte", color="amarillo",
        titular="El río Uruguay salió de su cauce y NISAR lo vio con el cielo cubierto",
        que_es="Estas son las pasadas reales del radar NISAR entre junio y octubre de 2026, una cada 12 días. Mové la línea de tiempo: el 12 de octubre aparecen manchas negras nuevas junto al río.",
        estado_detalle="En las zonas que NISAR marcó como posible agua nueva (celeste), el mapa de agua de NASA hecho con otro radar (Sentinel-1, 8 de octubre) pasó de 1 % a 53 % de agua; en las zonas sin cambio, de 0,3 % a 1,1 %. Es un indicio y no una validación: con una fecha sin crecida, la misma regla también marcó campos arados, apenas por encima de lo que habíamos fijado como tolerable (0,37 contra 0,33).",
        resultado="src/agua_crecida_uruguay.py (calibración relativa entre fechas): MARCADO = caída de HV ≥ 3 dB entre la referencia (25-ago y 6-sep) y el 12-oct, en celdas de 100 m. A1 1,1 % → 53,0 % (DSWx-S1, 8-oct), A2 0,3 % → 1,1 %, A3 2.114/5.765 = 0,367 (> 0,333: no pasa), A4 5.765 celdas.",
        evento="Crecida del río Uruguay, octubre de 2026 (alturas INA/CARU por encima del nivel de evacuación en Paso de los Libres)",
        limites=["Es un indicio, no una validación: el control temporal no pasó por poco.",
                 "El radar ve agua abierta; el agua debajo del monte de la costa puede no verse.",
                 "No mide la altura del río: ve dónde hay agua, no cuántos metros subió.",
                 "Entre una fecha y otra también cambia el color del campo (humedad, cosechas): eso no es agua."],
        capas=[dict(id="marcado", nombre="Agua nueva (12-oct)", archivos=marc, sobre="ojos",
                    texto_fecha="{f}: en celeste, las zonas donde NISAR vio el eco apagarse fuerte respecto de fines de agosto (solo se calcula para el 12 de octubre).",
                    leyenda={"titulo": "Posible agua nueva según NISAR (12 de octubre)", "gradiente": "linear-gradient(90deg,#40a8ff,#40a8ff)",
                             "izq": "", "der": "", "palabras": "<b style='color:#40a8ff'>Celeste</b>: el eco del radar se apagó fuerte (3 dB o más) respecto de fines de agosto. El mapa de agua de NASA confirma agua en la mitad de estas zonas; el resto incluye campos arados."})])
    escribir_punto("libres", k, fechas, b, od, extra)

def hacer_apipe():
    r = export_serie("apipe", ZONAS["apipe"])
    if not r: return
    k, fechas, b, od = r
    extra = dict(nombre="Yacyretá e isla Apipé", lugar="Río Paraná, Corrientes (Argentina) y Paraguay",
                 estado="Solo imagen, sin analizar", color="azul",
                 titular="La represa de Yacyretá y la isla Apipé, vistas por el radar",
                 que_es="Pasadas reales del radar NISAR entre junio y octubre de 2026. Se ven la represa, el embalse y los brazos del Paraná. El río venía creciendo y se esperaba que llegara al nivel de evacuación hacia el 20 de octubre.",
                 estado_detalle="Solo mostramos la imagen del radar: todavía no hicimos el análisis con confirmación independiente. Es la zona que vamos a seguir cuando el Paraná llegue a su pico.",
                 resultado="Composición src/radar_imagenes.py: R = HH, G = HV, B = HH/HV × brillo; escala fija en todas las fechas.",
                 evento="Crecida del río Paraná, octubre de 2026 (en curso)",
                 limites=["Es solo la imagen del radar: no afirma que haya agua nueva.", "Entre fechas cambia el color del campo (humedad, cultivos): eso no es agua."])
    escribir_punto("apipe", k, fechas, b, od, extra)

def hacer_la_guaira():
    pj = os.path.join(WEB, "la_guaira", "punto.json"); P = json.load(open(pj, encoding="utf-8"))
    cfg = ZONAS["la_guaira"]; b = P["bounds"]; bbox = (b["west"], b["south"], b["east"], b["north"])
    k, fechas = buscar(bbox, cfg["orbita"])
    if not fechas: print("  la_guaira: sin GCOV 126 D"); return
    elegir = lambda obj: min(fechas, key=lambda f: abs(int(f) - int(obj)))
    fa, fd = elegir(cfg["antes"]), elegir(cfg["despues"])
    print(f"  la_guaira: orbita {k} | antes {fa} | despues {fd}", flush=True)
    od = os.path.join(WEB, "la_guaira"); arch = {}
    for nombre, f in (("antes", fa), ("despues", fd)):
        d = leer(fechas[f], bbox)
        if d is None: print(f"   {f}: sin datos"); return
        guardar_png(imagen_lonlat(d, b), os.path.join(od, f"ojos_{f}.png")); arch[nombre] = f"ojos_{f}.png"
        print(f"   {nombre}: ojos_{f}.png", flush=True)
    capa = dict(id="ojos", nombre="Ojos de NISAR", momentos=arch, leyenda=LEYENDA_OJOS,
                texto_antes=f"Así vio el radar La Guaira y Caracas el {fecha_texto(fa)}, antes del sismo.",
                texto_despues=f"Y así el {fecha_texto(fd)}, después. A simple vista casi no cambia: el movimiento del suelo (medio metro) no se ve en la imagen, se mide comparando la fase de la onda entre dos pasadas.")
    P["capas"] = [c for c in P["capas"] if c["id"] != "ojos"] + [capa]
    json.dump(P, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  {pj}: capa 'ojos' agregada", flush=True)

if __name__ == "__main__":
    if "--prueba" in sys.argv:
        prueba(); sys.exit(0)
    print("Cargando librerias (1-3 min en Windows, no cortar)...", flush=True)
    import earthaccess, h5py, requests, pyproj, PIL
    print("Librerias cargadas.", flush=True)
    earthaccess.login(persist=True)
    pedidas = [a for a in sys.argv[1:] if a in ZONAS] or list(ZONAS)
    for z in pedidas:
        print(f"\n===== {z}", flush=True)
        {"libres": hacer_libres, "apipe": hacer_apipe, "la_guaira": hacer_la_guaira, "chaco": hacer_chaco}[z]()
    print("\nListo. Recargá el visor con Ctrl+F5.")
