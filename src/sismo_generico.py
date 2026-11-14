"""
sismo_generico.py -- LandSense
Analisis co-sismico GENERICO con NISAR a partir de SOLO el codigo del sismo en el USGS.
Nada escrito a mano para un lugar: zona, pares, controles y umbrales salen de reglas fijas.

Pasos:
  1. USGS: epicentro, magnitud, profundidad, fecha.
  2. Zona: cuadrado centrado en el epicentro, semilado = largo de ruptura estimado
     L = 10^(-2.44 + 0.59*M) km  (Wells & Coppersmith 1994, largo sub-superficial, todos
     los tipos de falla; BSSA 84(4):974-1002. Verificado: M6.8 -> 37.3 km), acotado a [25, 60] km.
  3. CMR directo: por cada (track, frame), el par que CRUZA la fecha (el mas corto) y su
     CONTROL = el par inmediatamente anterior (su fecha final = inicio del que cruza).
  4. Lectura por streaming de la fase desenrollada 80 m (mascara oficial, fase==0 fuera,
     coherencia >= 0.2, componentes conectados > 0, ionosfera restada). Mismo criterio que src/sismo_la_guaira_multiorbita.py.
  5. Descomposicion ESTE-OESTE / VERTICAL con un ascendente + un descendente y los vectores LOS
     reales (supuesto N = 0, como src/sismo_la_guaira_descomposicion.py). Referencia = anillo exterior de la zona (> 70% del
     semilado desde el epicentro).

Chequeos automaticos (UMBRALES FIJADOS ANTES de correr sobre un sismo nuevo):
  C1 senal vs control: rango p1-p99 del par que cruza >= 2 x el de su control (en al menos
     una orbita ascendente y una descendente). La Guaira dio 3.5-10x.
  C2 coincidencia: dos pares que cruzan con la MISMA direccion y de ORBITAS DISTINTAS (asc-asc o desc-desc)
     correlacion espacial r >= 0.7 donde se superponen. La Guaira dio 0.98.
     Si no hay dos de la misma direccion: "no evaluable" (no cuenta como aprobado).
  C3 control de la descomposicion: con los pares de control, |E| y |U| p99 < 1/2 de los del sismo.
AJUSTES (decididos despues de correr la prueba de regresion con La Guaira y el caso Kumamoto):
  - Componente conectado: por par, solo el componente del desenrollado con mas pixeles validos a
    menos de 0.5 x semilado del epicentro (las islas separadas por agua tienen desfase arbitrario).
  - C2 solo entre frames que pasan C1 por si solos.
  - C3 frente a la componente DOMINANTE del sismo; la otra se informa "no detectable" si su p99
    no supera 2 x la del control. C3 "no evaluable" si hay < 2000 px comunes asc+desc a menos de
    0.5 x semilado del epicentro (en sismo o control).
  Umbrales numericos (2x, 0.7, 0.5) SIN cambios. Criterio de exito fijado antes de re-correr:
  La Guaira 3 de 3; Kumamoto no debe aprobar una descomposicion que dependa de islas.
Estado automatico: "SIN VERIFICAR - paso N de 3 chequeos". Nunca "validado": eso requiere
comparar con una fuente independiente (lo hace una persona).

Uso:
    python src/sismo_generico.py us6000tgb9
    python src/sismo_generico.py us6000u43g --con-ur   (suma la coleccion Urgent Response)
    (Kumamoto 2026). Prueba de regresion: us6000t7zp (La Guaira) debe dar ~oeste al sur del aeropuerto.
Salidas: ../data/processed/sismos/<id>/informe.txt, mapas.png, EU.npz, resumen.json
"""
import sys
print("Iniciando analisis generico de sismo... (cargar librerias puede tardar 1-2 min)", flush=True)
import os, re, time, json
from datetime import datetime, timedelta, timezone
import numpy as np
import requests
import h5py
import earthaccess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer, Geod
print("Librerias cargadas.", flush=True)

EVENTO = sys.argv[1] if len(sys.argv) > 1 else "us6000tgb9"
COLECCION = "C2854335566-ASF"          # NISAR_L2_GUNW_PROVISIONAL_V1
# Opcion --con-ur suma la coleccion Urgent Response (NISAR_UR_L2: mismo producto GUNW,
# misma version de procesamiento P05023). Sin la opcion, usa solo PROVISIONAL.
COLECCION_UR = "C2887502657-ASF"
COLECCIONES = [COLECCION] + ([COLECCION_UR] if "--con-ur" in sys.argv else [])
ROOT = "science/LSAR/GUNW/grids/frequencyA"
UNW = f"{ROOT}/unwrappedInterferogram"
RG = "science/LSAR/GUNW/metadata/radarGrid"
C_LUZ = 299792458.0
UMBRAL_C1, UMBRAL_C2, UMBRAL_C3 = 2.0, 0.7, 0.5
ALTURA = 500.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "sismos", EVENTO)
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "informe.txt"), "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---------- 1) USGS ----------
ev = requests.get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={EVENTO}&format=geojson",
                  timeout=60).json()
lon0, lat0, prof = ev["geometry"]["coordinates"]
mag = float(ev["properties"]["mag"])
t0 = datetime.fromtimestamp(ev["properties"]["time"] / 1000, timezone.utc)
FECHA = t0.strftime("%Y%m%d")
p(f"Sismo {EVENTO}: {ev['properties']['place']} | M{mag} | prof {prof} km | {t0:%Y-%m-%d %H:%M} UTC")
if prof > 30:
    p(f"AVISO: profundidad {prof} km > 30 km: la deformacion en superficie puede ser muy chica (ver Colombia).")

# ---------- 2) Zona ----------
L = 10 ** (-2.44 + 0.59 * mag)
R_KM = float(np.clip(L, 25, 60))
dlat = R_KM / 111.0; dlon = R_KM / (111.0 * np.cos(np.radians(lat0)))
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat
p(f"Zona: semilado {R_KM:.0f} km (largo de ruptura estimado {L:.0f} km) | lon {LON_MIN:.2f}..{LON_MAX:.2f} lat {LAT_MIN:.2f}..{LAT_MAX:.2f}")

# ---------- 3) Pares (CMR directo) ----------
ini = (t0 - timedelta(days=60)).strftime("%Y-%m-%dT00:00:00Z"); fin = (t0 + timedelta(days=45)).strftime("%Y-%m-%dT00:00:00Z")
gran = []
vistos = set()
entradas = []
for col in COLECCIONES:
    url = (f"https://cmr.earthdata.nasa.gov/search/granules.json?collection_concept_id={col}"
           f"&bounding_box={LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}&temporal={ini},{fin}&page_size=500")
    entradas += [(col, e) for e in requests.get(url, timeout=60).json()["feed"]["entry"]]
for col, e in entradas:
    m = re.search(r"GUNW_\d{3}_(\d{3})_([AD])_(\d{3})_\d{3}_\d{4}_SH_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{8})T", e["title"])
    h5 = [l["href"] for l in e.get("links", []) if l["href"].startswith("https") and l["href"].endswith(".h5")]
    clave_par = (m.group(1), m.group(3), m.group(4), m.group(5)) if m else None
    if m and h5 and clave_par not in vistos:          # si un par esta en PR y UR, queda el de PR (va primero)
        vistos.add(clave_par)
        gran.append(dict(track=m.group(1), dir=m.group(2), frame=m.group(3), ref=m.group(4), sec=m.group(5), url=h5[0],
                         col="UR" if col == COLECCION_UR else "PR"))
dias = lambda g: (datetime.strptime(g["sec"], "%Y%m%d") - datetime.strptime(g["ref"], "%Y%m%d")).days
grupos = {}
for g in gran:
    if g["ref"] <= FECHA <= g["sec"]:
        k = (g["track"], g["frame"])
        if k not in grupos or dias(g) < dias(grupos[k]):
            grupos[k] = g
p(f"Granulos GUNW: {len(gran)} | (track, frame) con par que cruza: {len(grupos)}")
plan = []
for k, g in sorted(grupos.items()):
    ctrl = [c for c in gran if (c["track"], c["frame"]) == k and c["sec"] == g["ref"]]
    p(f"  track {g['track']} {g['dir']} frame {g['frame']}: cruza {g['ref']}->{g['sec']} ({dias(g)} d, {g['col']}) | "
      f"control: {ctrl[0]['ref'] + '->' + ctrl[0]['sec'] + f' ({dias(ctrl[0])} d)' if ctrl else 'NO HAY'}")
    plan.append(dict(cruza=g, control=ctrl[0] if ctrl else None))
if not plan:
    p("RESULTADO: no hay pares NISAR que crucen el sismo sobre la zona. Nada que analizar."); sys.exit(0)

# ---------- 4) Lectura ----------
earthaccess.login(persist=True)
def mascara_oficial(mk):
    m8 = np.where(mk == 255, 0, mk & 0xFF)
    return (mk != 255) & ((m8 // 10) % 10 > 0) & (m8 % 10 > 0) & ((m8 // 100) != 1)

def leer(g):
    for intento in range(1, 5):
        try:
            with h5py.File(earthaccess.open([g["url"]])[0], "r") as f:
                lam = C_LUZ / float(f[f"{ROOT}/centerFrequency"][()])
                x = f[f"{UNW}/xCoordinates"][:]; y = f[f"{UNW}/yCoordinates"][:]
                epsg = int(f[f"{UNW}/projection"][()])
                tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                xa, ya = tr.transform(LON_MIN, LAT_MIN); xb, yb = tr.transform(LON_MAX, LAT_MAX)
                c = np.where((x >= min(xa, xb)) & (x <= max(xa, xb)))[0]
                r = np.where((y >= min(ya, yb)) & (y <= max(ya, yb)))[0]
                if c.size < 10 or r.size < 10:
                    return None
                r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
                ph = f[f"{UNW}/HH/unwrappedPhase"][r0:r1, c0:c1].astype(np.float32)
                io = f[f"{UNW}/HH/ionospherePhaseScreen"][r0:r1, c0:c1].astype(np.float32)
                co = f[f"{UNW}/HH/coherenceMagnitude"][r0:r1, c0:c1].astype(np.float32)
                cc = f[f"{UNW}/HH/connectedComponents"][r0:r1, c0:c1].astype(np.int32)
                mk = f[f"{UNW}/mask"][r0:r1, c0:c1].astype(np.int32)
                hh = f[f"{RG}/heightAboveEllipsoid"][:]; kk = int(np.abs(hh - ALTURA).argmin())
                V = dict(e=f[f"{RG}/losUnitVectorX"][kk].astype(np.float64), n=f[f"{RG}/losUnitVectorY"][kk].astype(np.float64),
                         x=f[f"{RG}/xCoordinates"][:], y=f[f"{RG}/yCoordinates"][:], epsg=int(f[f"{RG}/projection"][()]))
            V["u"] = np.sqrt(np.clip(1 - V["e"] ** 2 - V["n"] ** 2, 0, 1))
            ok = np.isfinite(ph) & np.isfinite(io) & (co >= 0.2) & (cc > 0) & mascara_oficial(mk) & (ph != 0)
            # componente conectado dominante cerca del epicentro (islas = desfase arbitrario)
            XX, YY = np.meshgrid(x[c0:c1], y[r0:r1])
            lo_, la_ = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(XX, YY)
            dkm = Geod(ellps="WGS84").inv(np.full(lo_.shape, lon0), np.full(la_.shape, lat0), lo_, la_)[2] / 1000
            cerca = ok & (dkm < 0.5 * R_KM)
            if cerca.sum() > 0:
                labs, cnt = np.unique(cc[cerca], return_counts=True)
                comp = int(labs[cnt.argmax()])
                n_otros = int((ok & (cc != comp)).sum())
                ok &= cc == comp
                p(f"    componente conectado {comp} (cerca del epicentro); descartados {n_otros} px de otros componentes")
            else:
                p("    sin pixeles validos cerca del epicentro en este par")
            los = np.where(ok, -(ph - io) * lam / (4 * np.pi) * 100, np.nan)   # cm, + = hacia el satelite
            if np.isfinite(los).sum() < 1000:
                return None
            los -= np.nanmedian(los)
            return dict(los=los, x=x[c0:c1], y=y[r0:r1], epsg=epsg, lam=lam, V=V)
        except Exception as ex:
            p(f"    reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})"); time.sleep(6)
    return None

def rango(a):
    return float(np.nanpercentile(a, 99) - np.nanpercentile(a, 1))

for item in plan:
    g = item["cruza"]
    p(f"\n--- track {g['track']} {g['dir']} frame {g['frame']} ---")
    item["d"] = leer(g)
    item["dc"] = leer(item["control"]) if item["control"] else None
    if item["d"] is None:
        p("  par que cruza: sin datos validos en la zona"); continue
    item["rango"] = rango(item["d"]["los"])
    txt = f"  cruza: validos {int(np.isfinite(item['d']['los']).sum())}, rango p1-p99 {item['rango']:.1f} cm (lambda {item['d']['lam']:.4f} m)"
    if item["dc"] is not None:
        item["rango_c"] = rango(item["dc"]["los"]); item["ratio"] = item["rango"] / max(item["rango_c"], 0.1)
        txt += f" | control rango {item['rango_c']:.1f} cm -> senal/control = {item['ratio']:.1f}x"
    if item["dc"] is not None and dias(g) > dias(item["control"]):
        txt += f"\n  (ojo: el par que cruza dura {dias(g)} d y el control {dias(item['control'])} d; mas dias = mas ruido posible)"
    p(txt)

validos = [i for i in plan if i.get("d") is not None]

# ---------- guardado de los mapas LOS (no cambia ningun criterio) ----------
_guardar = {}
for i in plan:
    for tipo, dd in (("cruza", i.get("d")), ("control", i.get("dc"))):
        if dd is None: continue
        g_ = i["cruza"] if tipo == "cruza" else i["control"]
        k = f"{g_['track']}{g_['dir']}_{g_['frame']}_{tipo}"
        _guardar.update({f"{k}__los": dd["los"].astype(np.float32), f"{k}__x": dd["x"], f"{k}__y": dd["y"],
                         f"{k}__epsg": np.array(dd["epsg"]), f"{k}__lam": np.array(dd["lam"]),
                         f"{k}__fechas": np.array(f"{g_['ref']}-{g_['sec']}"),
                         f"{k}__Ve": dd["V"]["e"], f"{k}__Vn": dd["V"]["n"], f"{k}__Vu": dd["V"]["u"],
                         f"{k}__Vx": dd["V"]["x"], f"{k}__Vy": dd["V"]["y"], f"{k}__Vepsg": np.array(dd["V"]["epsg"])})
if _guardar:
    np.savez_compressed(os.path.join(OUT, "pares_los.npz"), **_guardar)
    p(f"Mapas LOS guardados: {os.path.join(OUT, 'pares_los.npz')} ({len(_guardar)//12} pares)")

# ---------- C1 ----------
c1 = {d: any(i.get("ratio", 0) >= UMBRAL_C1 for i in validos if i["cruza"]["dir"] == d) for d in "AD"}
C1 = c1["A"] and c1["D"]
p(f"\nC1 senal vs control (>= {UMBRAL_C1}x en asc y en desc): asc {c1['A']}, desc {c1['D']} -> {'PASA' if C1 else 'NO PASA'}")

# ---------- C2 ----------
def muestrear(campo, gx, gy, gepsg, X, Y, epsg_dest):
    if gepsg != epsg_dest:
        X, Y = Transformer.from_crs(f"EPSG:{epsg_dest}", f"EPSG:{gepsg}", always_xy=True).transform(X, Y)
    ix = np.round((X - gx[0]) / (gx[1] - gx[0])).astype(int); iy = np.round((Y - gy[0]) / (gy[1] - gy[0])).astype(int)
    ok = (ix >= 0) & (ix < len(gx)) & (iy >= 0) & (iy < len(gy))
    out = np.full(X.shape, np.nan); out[ok] = campo[iy[ok], ix[ok]]
    return out

C2, C2_eval = False, False
for d in "AD":
    ms = [i for i in validos if i["cruza"]["dir"] == d and i.get("ratio", 0) >= UMBRAL_C1]
    for a_i in range(len(ms)):
        for b_i in range(a_i + 1, len(ms)):
            if ms[a_i]["cruza"]["track"] == ms[b_i]["cruza"]["track"]:
                continue        # mismo track (frames vecinos de la MISMA pasada): no es independiente
            A, B = ms[a_i]["d"], ms[b_i]["d"]
            X, Y = np.meshgrid(A["x"], A["y"])
            Bg = muestrear(B["los"], B["x"], B["y"], B["epsg"], X, Y, A["epsg"])
            com = np.isfinite(A["los"]) & np.isfinite(Bg)
            if com.sum() < 2000:
                continue
            C2_eval = True
            r_ = float(np.corrcoef(A["los"][com] - np.median(A["los"][com]), Bg[com] - np.median(Bg[com]))[0, 1])
            p(f"C2 {d}: track {ms[a_i]['cruza']['track']} vs {ms[b_i]['cruza']['track']}: r = {r_:.2f} ({int(com.sum())} px en comun)")
            C2 = C2 or r_ >= UMBRAL_C2
p(f"C2 coincidencia misma direccion (r >= {UMBRAL_C2}): {'PASA' if C2 else ('NO PASA' if C2_eval else 'NO EVALUABLE (no hay dos pares de la misma direccion superpuestos)')}")

# ---------- 5) Descomposicion + C3 ----------
asc = sorted([i for i in validos if i["cruza"]["dir"] == "A"], key=lambda i: -i.get("ratio", 0))
des = sorted([i for i in validos if i["cruza"]["dir"] == "D"], key=lambda i: -i.get("ratio", 0))
C3 = False; C3_eval = False; R = {}
if asc and des:
    ia, idd = asc[0], des[0]
    p(f"\nDescomposicion con asc {ia['cruza']['track']} + desc {idd['cruza']['track']} (los de mayor senal/control)")
    def descomp(dA_, dD_):
        A, D = dA_, dD_
        X, Y = np.meshgrid(A["x"], A["y"]); ep = A["epsg"]
        la = A["los"]; ld = muestrear(D["los"], D["x"], D["y"], D["epsg"], X, Y, ep)
        eA = muestrear(ia["d"]["V"]["e"], ia["d"]["V"]["x"], ia["d"]["V"]["y"], ia["d"]["V"]["epsg"], X, Y, ep)
        uA = muestrear(ia["d"]["V"]["u"], ia["d"]["V"]["x"], ia["d"]["V"]["y"], ia["d"]["V"]["epsg"], X, Y, ep)
        eD = muestrear(idd["d"]["V"]["e"], idd["d"]["V"]["x"], idd["d"]["V"]["y"], idd["d"]["V"]["epsg"], X, Y, ep)
        uD = muestrear(idd["d"]["V"]["u"], idd["d"]["V"]["x"], idd["d"]["V"]["y"], idd["d"]["V"]["epsg"], X, Y, ep)
        lon, lat = Transformer.from_crs(f"EPSG:{ep}", "EPSG:4326", always_xy=True).transform(X, Y)
        dist = Geod(ellps="WGS84").inv(np.full(lon.shape, lon0), np.full(lat.shape, lat0), lon, lat)[2] / 1000
        com = np.isfinite(la) & np.isfinite(ld) & np.isfinite(eA) & np.isfinite(eD)
        ref = com & (dist > 0.7 * R_KM)
        if ref.sum() < 2000:
            ref = com
        la = la - np.nanmedian(la[ref]); ld = ld - np.nanmedian(ld[ref])
        det = eA * uD - eD * uA; bien = com & (np.abs(det) > 0.2)
        E = np.where(bien, (la * uD - ld * uA) / det, np.nan); U = np.where(bien, (eA * ld - eD * la) / det, np.nan)
        cerca_px = int((bien & (dist < 0.5 * R_KM)).sum())
        return dict(E=E, U=U, x=A["x"], y=A["y"], epsg=ep, dist=dist, ref_px=int(ref.sum()), cerca_px=cerca_px)
    R["CRUZA"] = descomp(ia["d"], idd["d"])
    for k in ["CRUZA"]:
        E, U = R[k]["E"], R[k]["U"]
        p(f"  {k}: E p1 {np.nanpercentile(E,1):+.1f} / p99 {np.nanpercentile(E,99):+.1f} cm | U p1 {np.nanpercentile(U,1):+.1f} / p99 {np.nanpercentile(U,99):+.1f} cm | referencia {R[k]['ref_px']} px")
    if ia["dc"] is not None and idd["dc"] is not None:
        R["CONTROL"] = descomp(ia["dc"], idd["dc"])
        Ec, Uc = R["CONTROL"]["E"], R["CONTROL"]["U"]
        p(f"  CONTROL: E p1 {np.nanpercentile(Ec,1):+.1f} / p99 {np.nanpercentile(Ec,99):+.1f} cm | U p1 {np.nanpercentile(Uc,1):+.1f} / p99 {np.nanpercentile(Uc,99):+.1f} cm")
        amp = lambda a: float(np.nanpercentile(np.abs(a), 99)) if np.isfinite(a).sum() > 100 else float("nan")
        cp_s, cp_c = R["CRUZA"]["cerca_px"], R["CONTROL"]["cerca_px"]
        p(f"  px comunes asc+desc a < {0.5*R_KM:.0f} km del epicentro: sismo {cp_s}, control {cp_c}")
        if min(cp_s, cp_c) < 2000:
            C3_eval = False
            p("C3: NO EVALUABLE (poco dato comun asc+desc cerca del epicentro: la descomposicion no representa la zona del sismo)")
        else:
            C3_eval = True
            aS = {c: amp(R["CRUZA"][c]) for c in "EU"}; aC = {c: amp(R["CONTROL"][c]) for c in "EU"}
            dom = max("EU", key=lambda c: aS[c]); otra = "U" if dom == "E" else "E"
            r_dom = aC[dom] / max(aS[dom], 0.1)
            C3 = r_dom < UMBRAL_C3
            p(f"C3 control vs componente dominante ({dom}): control p99 {aC[dom]:.1f} / sismo p99 {aS[dom]:.1f} cm = {r_dom:.2f} "
              f"(< {UMBRAL_C3}) -> {'PASA' if C3 else 'NO PASA'}")
            det_otra = aS[otra] > 2 * aC[otra]
            p(f"  componente {otra}: sismo p99 {aS[otra]:.1f} vs control {aC[otra]:.1f} cm -> "
              f"{'detectable' if det_otra else 'NO DETECTABLE (dentro del ruido del control)'}")
    else:
        p("C3: falta un par de control asc o desc -> NO EVALUABLE")
    near = R["CRUZA"]["dist"] < 0.3 * R_KM
    E, U = R["CRUZA"]["E"], R["CRUZA"]["U"]
    if np.isfinite(E[near]).sum() > 100:
        p(f"  Cerca del epicentro (< {0.3*R_KM:.0f} km): E p5 {np.nanpercentile(E[near],5):+.1f} / p95 {np.nanpercentile(E[near],95):+.1f} cm | "
          f"U p5 {np.nanpercentile(U[near],5):+.1f} / p95 {np.nanpercentile(U[near],95):+.1f} cm")
    np.savez_compressed(os.path.join(OUT, "EU.npz"), E_cm=E.astype(np.float32), U_cm=U.astype(np.float32),
                        x=R["CRUZA"]["x"], y=R["CRUZA"]["y"], epsg=R["CRUZA"]["epsg"])
else:
    p("\nNo hay ascendente Y descendente validos: no se descompone (C3 no evaluable).")

n_ok = int(C1) + int(C2) + int(C3)
estado = f"SIN VERIFICAR - paso {n_ok} de 3 chequeos automaticos"
p(f"\nESTADO AUTOMATICO: {estado}")
p("  (Validar requiere comparar con una fuente independiente: GPS, agencia oficial, NASA. Lo hace una persona.)")
json.dump(dict(evento=EVENTO, lugar=ev["properties"]["place"], mag=mag, prof_km=prof, fecha=FECHA, semilado_km=R_KM,
               pares=[dict(track=i["cruza"]["track"], dir=i["cruza"]["dir"], cruza=f"{i['cruza']['ref']}-{i['cruza']['sec']}",
                           ratio=i.get("ratio")) for i in validos],
               C1=C1, C2=C2 if C2_eval else None, C3=C3 if C3_eval else None, estado=estado),
          open(os.path.join(OUT, "resumen.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- figura ----------
if not validos:
    p("Sin pares con datos validos en la zona: no hay figura.")
    raise SystemExit(0)
paneles = [(f"{i['cruza']['track']} {i['cruza']['dir']} CRUZA {i['cruza']['ref']}->{i['cruza']['sec']}", i["d"]) for i in validos]
paneles += [(f"{i['cruza']['track']} {i['cruza']['dir']} CONTROL", i["dc"]) for i in validos if i["dc"] is not None]
n = len(paneles) + (2 if R else 0); cols = 3; filas = int(np.ceil(n / cols))
fig, ax = plt.subplots(filas, cols, figsize=(17, 4.6 * filas), squeeze=False)
_rs = [i["rango"] for i in validos if (i.get("ratio") or 0) >= 2] or [i["rango"] for i in validos]   # solo escala de colores
vm = max(5.0, float(np.nanpercentile(_rs, 50)) / 2)
k = 0
for tit, d in paneles:
    a = ax[k // cols, k % cols]; ext = [d["x"][0], d["x"][-1], d["y"][-1], d["y"][0]]
    im = a.imshow(d["los"], cmap="RdBu_r", vmin=-vm, vmax=vm, extent=ext)
    ex, ey = Transformer.from_crs("EPSG:4326", f"EPSG:{d['epsg']}", always_xy=True).transform(lon0, lat0)
    a.plot(ex, ey, "k*", ms=12); a.set_title(tit + "\nLOS cm (+ hacia el satelite), * epicentro", fontsize=8)
    plt.colorbar(im, ax=a); k += 1
if R:
    for nom, comp in [("ESTE-OESTE (rojo = este)", "E"), ("VERTICAL (rojo = sube)", "U")]:
        a = ax[k // cols, k % cols]; r = R["CRUZA"]; ext = [r["x"][0], r["x"][-1], r["y"][-1], r["y"][0]]
        im = a.imshow(r[comp], cmap="RdBu_r", vmin=-vm, vmax=vm, extent=ext); a.set_title(nom + " cm", fontsize=9)
        ex, ey = Transformer.from_crs("EPSG:4326", f"EPSG:{r['epsg']}", always_xy=True).transform(lon0, lat0)
        a.plot(ex, ey, "k*", ms=12); plt.colorbar(im, ax=a); k += 1
for j in range(k, filas * cols):
    ax[j // cols, j % cols].axis("off")
plt.tight_layout(); fig.savefig(os.path.join(OUT, "mapas.png"), dpi=90)
p(f"\nFigura: {os.path.join(OUT, 'mapas.png')}")
LOG.close()
