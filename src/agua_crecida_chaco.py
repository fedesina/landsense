"""
agua_crecida_chaco.py -- LandSense. Misma receta y criterios A1-A4 que src/agua_crecida_uruguay.py; solo cambian ZONAS, fechas y carpeta de salida.
Prueba CIEGA: crecida Parana/Paraguay en el Gran Resistencia, oct-2026 (durante El Nino). Zona, fechas y criterios fijados antes de mirar los datos NISAR.
(descripcion de la receta, la misma de src/agua_crecida_uruguay.py)
Receta con calibracion relativa entre fechas, como src/agua_gcov_exploratorio.py.
REPLICA de la receta del Delta del Parana (src/agua_gcov_exploratorio.py y src/agua_delta_*.py) en la crecida del rio Uruguay de octubre 2026.
Criterios fijados ANTES de ver datos NISAR (parametros abajo).

Receta (sin cambios respecto del Delta):
  - GCOV DHDH, HV promediado a ~100 m (potencia lineal).
  - MARCADO = HV(evento)/HV(referencia) <= -3 dB ; CONTROL = |cambio| < 1 dB.
  - Se excluyen celdas que ya eran agua (DSWx >= 50 % en la ventana de referencia) o sin dato DSWx ahi.
Verdad independiente: OPERA DSWx-HLS (optico, agua = 1,2) y DSWx-S1 (radar banda C, agua = 1).
Criterios por orbita:
  A1 MARCADO: % agua DSWx evento - pre >= +20 puntos (al menos un producto)
  A2 mismo producto, CONTROL: |evento - pre| <= 5 puntos
  A3 control temporal: celdas marcadas con la fecha control <= 1/3 de las del evento
  A4 MARCADO >= 50 celdas (si no: NO EVALUABLE)
  PASA = A1 y A2 y A3 y A4. Una orbita -> VERIFICADO; dos orbitas en la misma zona -> VALIDADO.

Uso:
    python src/agua_crecida_uruguay.py            (todas las zonas)
    python src/agua_crecida_uruguay.py concordia  (una zona)
    python src/agua_crecida_uruguay.py --prueba   (datos simulados: caso bueno PASA, caso ruido FALLA; sin internet)
Salidas: ../data/processed/crecida/<zona>_<track>.png, <zona>_<track>.npz, informe.txt, resumen.json
Login Earthdata: tu propio usuario (el nombre de usuario, NO el email).
"""
import os, re, sys, json, time
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

# ---------------- parametros (fijados antes de mirar los datos) ----------------
GCOV = "C2854338529-ASF"
DSWX = {"HLS": ("C2617126679-POCLOUD", [1, 2]), "S1": ("C2949811996-POCLOUD", [1])}
G = "science/LSAR/GCOV/grids/frequencyA"
ZONAS = {"gran_resistencia": (-59.05, -27.60, -58.55, -27.00)}   # Barranqueras, Vilelas, Isla del Cerrito, Las Palmas, Corrientes
DESDE, HASTA, EVENTO_MIN = "20260815", "20261024", "20261016"   # fijado ANTES de ver datos NISAR de la zona
OBJ_M = 100.0
UMB_MARCA, UMB_CTRL = -3.0, 1.0
A1_PP, A2_PP, A3_FRAC, A4_MIN = 20.0, 5.0, 1 / 3, 50
MIN_VALIDO, AGUA_PREVIA = 0.5, 0.5
VENT_REF, VENT_EV = 5, (-4, 3)

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "data", "processed", "crecida", "chaco")
os.makedirs(OUT, exist_ok=True)
LOG = None
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    if LOG: LOG.write(s + "\n"); LOG.flush()

def fd(s): return datetime.strptime(s, "%Y%m%d")

# ---------------- logica pura (probada con --prueba) ----------------
def bloques(a, k):
    ny, nx = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:ny, :nx].reshape(ny // k, k, nx // k, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))

def elegir_fechas(fechas):
    """Regla fijada: ref = 2 primeras, control = 3.a, evento = ultima (<= HASTA). None si no cumple."""
    f = sorted(x for x in fechas if DESDE <= x <= HASTA)
    if len(f) < 4 or f[-1] < EVENTO_MIN:
        return None
    return dict(ref=f[:2], control=f[2], evento=f[-1], todas=f)

def cambios(HV, fechas, sel):
    ref = np.nanmean(np.stack([HV[fechas.index(x)] for x in sel["ref"]]), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_ev = 10 * np.log10(HV[fechas.index(sel["evento"])] / ref)
        d_ct = 10 * np.log10(HV[fechas.index(sel["control"])] / ref)
    return ref, d_ev, d_ct

def grupos(d_ev, d_ct, excluir):
    ok = np.isfinite(d_ev) & ~excluir
    M = ok & (d_ev <= UMB_MARCA)
    C = ok & (np.abs(d_ev) < UMB_CTRL)
    Mc = np.isfinite(d_ct) & ~excluir & (d_ct <= UMB_MARCA)
    return M, C, Mc

def evaluar(nM, nMc, verdad):
    """verdad = {producto: dict(pre_M, pre_C, ev_M, ev_C) o None}. Devuelve dict con A1-A4 y estado."""
    r = dict(nM=int(nM), nMc=int(nMc), A4=nM >= A4_MIN, A3=nMc <= A3_FRAC * nM, prod=None, A1=False, A2=False)
    for prod, v in verdad.items():
        if v is None:
            continue
        a1 = (v["ev_M"] - v["pre_M"]) >= A1_PP
        a2 = abs(v["ev_C"] - v["pre_C"]) <= A2_PP
        if a1 and a2:
            r.update(prod=prod, A1=True, A2=True); break
        if a1 and r["prod"] is None:
            r.update(prod=prod, A1=True, A2=False)
    if not r["A4"]:
        r["estado"] = "NO EVALUABLE (pocas celdas marcadas)"
    elif r["A1"] and r["A2"] and r["A3"]:
        r["estado"] = "PASA"
    else:
        r["estado"] = "NO PASA"
    return r

def resumen_dswx(serie, sel):
    """serie = lista de dict(fecha, validos, M, C, agua_frac_por_celda) -> dict pre/ev o None."""
    r0 = fd(sel["ref"][0]) - timedelta(days=VENT_REF); r1 = fd(sel["ref"][1]) + timedelta(days=VENT_REF)
    ev = fd(sel["evento"])
    pre = [s for s in serie if r0 <= fd(s["fecha"]) <= r1 and s["validos"] >= MIN_VALIDO]
    evs = [s for s in serie if ev + timedelta(days=VENT_EV[0]) <= fd(s["fecha"]) <= ev + timedelta(days=VENT_EV[1])
           and s["validos"] >= MIN_VALIDO]
    if not pre or not evs:
        return None
    e = min(evs, key=lambda s: (abs((fd(s["fecha"]) - ev).days), s["fecha"]))
    return dict(pre_M=float(np.mean([s["M"] for s in pre])), pre_C=float(np.mean([s["C"] for s in pre])),
                ev_M=e["M"], ev_C=e["C"], fecha_ev=e["fecha"], fechas_pre=[s["fecha"] for s in pre])

# ---------------- prueba con datos simulados ----------------
def prueba():
    rng = np.random.default_rng(1)
    fechas = ["20260824", "20260905", "20260917", "20260929", "20261011"]
    sel = elegir_fechas(fechas); assert sel["control"] == "20260917" and sel["evento"] == "20261011"
    assert elegir_fechas(fechas[:4]) is None  # sin fecha de evento >= 08-oct
    ny = nx = 60
    for caso in ("bueno", "ruido", "falsa"):
        base = 10 ** (rng.normal(-16, 1.5, (ny, nx)) / 10)
        HV = np.stack([base * 10 ** (rng.normal(0, 0.5, (ny, nx)) / 10) for _ in fechas])
        inund = np.zeros((ny, nx), bool); inund[20:35, 10:50] = True     # 600 celdas
        rio = np.zeros((ny, nx), bool); rio[:, 55:] = True               # agua permanente
        if caso in ("bueno", "falsa"):
            HV[-1][inund] *= 10 ** (-7 / 10)   # "falsa": baja el HV pero NO hay agua (ej. desmonte)
        HV[:, rio] = 10 ** (-28 / 10)
        ref, d_ev, d_ct = cambios(HV, fechas, sel)
        M, C, Mc = grupos(d_ev, d_ct, rio)
        # verdad simulada: en el caso bueno, el agua aparece en 'inund' solo el 11-oct
        def pct(mask, agua): return 100 * agua[mask].mean() if mask.sum() else 0.0
        serie = []
        for f in ["20260822", "20260901", "20260915", "20260930", "20261009", "20261011"]:
            agua = rio.copy()
            if caso == "bueno" and f >= "20261009":
                agua = agua | inund
            serie.append(dict(fecha=f, validos=0.9, M=pct(M, agua), C=pct(C, agua)))
        v = resumen_dswx(serie, sel)
        r = evaluar(M.sum(), Mc.sum(), {"HLS": v, "S1": v})
        print(f"[prueba] caso {caso}: {r}")
        assert (r["estado"] == "PASA") == (caso == "bueno"), "la logica no separa los casos"
        assert not (M & rio).any(), "el rio permanente no debe marcarse"
    # a_celdas: raster DSWx de 30 m sobre grilla de 100 m (2x2 celdas), agua en la mitad izquierda
    class T: pass
    tr = T(); tr.c, tr.a, tr.f, tr.e = 0.0, 30.0, 0.0, -30.0
    a = np.zeros((7, 7), np.uint8); a[:, :3] = 1; a[0, 6] = 253
    ag, va = a_celdas(a, tr, [1], 0.0, (2, 2, 0.0, 0.0, 100.0, -100.0))
    assert ag[0, 0] == 9 and ag[0, 1] == 0 and va[0, 1] == 8, (ag, va)
    print("[prueba] OK: caso bueno PASA, caso ruido no pasa, agua permanente excluida.")

# ---------------- datos reales ----------------
def buscar_cmr(cid, bbox, t0, t1):
    import requests
    w, s, e, n = bbox; out, page = [], 1
    while True:
        r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", timeout=60, params=dict(
            collection_concept_id=cid, bounding_box=f"{w},{s},{e},{n}", temporal=f"{t0},{t1}",
            page_size=200, page_num=page)).json()["feed"]["entry"]
        out += r
        if len(r) < 200: return out
        page += 1

def gcov_por_orbita(bbox):
    g = defaultdict(dict)
    for ent in buscar_cmr(GCOV, bbox, f"{DESDE[:4]}-{DESDE[4:6]}-{DESDE[6:]}T00:00:00Z", f"{HASTA[:4]}-{HASTA[4:6]}-{HASTA[6:]}T23:59:59Z"):
        m = re.search(r"GCOV_\d{3}_(\d{3})_([AD])_(\d{3})_\d{4}_(\w{4})_[A-Z]_(\d{8})T.*_(\d{3})$", ent["title"])
        h5 = [l["href"] for l in ent.get("links", []) if l["href"].startswith("https") and l["href"].endswith(".h5")]
        if m and h5 and m.group(4) == "DHDH":
            k = (m.group(1), m.group(2), m.group(3)); f = m.group(5); ver = m.group(6)
            if f not in g[k] or ver > g[k][f][0]:          # version mas alta de cada fecha
                g[k][f] = (ver, h5[0])
    return {k: {f: u for f, (v, u) in d.items()} for k, d in g.items()}

def leer_gcov(u, bbox):
    import h5py, earthaccess
    from pyproj import Transformer
    w, s, e, n = bbox
    for intento in range(1, 5):
        try:
            with h5py.File(earthaccess.open([u])[0], "r") as f:
                x = f[f"{G}/xCoordinates"][:]; y = f[f"{G}/yCoordinates"][:]
                epsg = int(f[f"{G}/projection"][()])
                tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                xa, ya = tr.transform(w, s); xb, yb = tr.transform(e, n)
                c = np.where((x >= min(xa, xb)) & (x <= max(xa, xb)))[0]
                r = np.where((y >= min(ya, yb)) & (y <= max(ya, yb)))[0]
                if c.size < 50 or r.size < 50: return None
                r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
                hh = f[f"{G}/HHHH"][r0:r1, c0:c1].astype(np.float32)
                hv = f[f"{G}/HVHV"][r0:r1, c0:c1].astype(np.float32)
            paso = abs(float(x[1] - x[0])); k = max(1, int(round(OBJ_M / paso)))
            bad = ~(np.isfinite(hh) & np.isfinite(hv) & (hh > 0) & (hv > 0)); hh[bad] = np.nan; hv[bad] = np.nan
            return dict(hh=bloques(hh, k), hv=bloques(hv, k), paso=paso, k=k, epsg=epsg,
                        x0=float(x[c0]), y0=float(y[r0]), dx=float(x[1] - x[0]) * k, dy=float(y[1] - y[0]) * k)
        except Exception as ex:
            p(f"    reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})"); time.sleep(6)
    return None


def dswx_por_fecha(cid, bbox, t0, t1):
    mejor = {}
    for e in buscar_cmr(cid, bbox, t0, t1):
        m = re.search(r"_T(\w{5})_(\d{8})T\d{6}Z_(\d{8}T\d{6})Z", e["title"])
        url = [l["href"] for l in e["links"] if l["href"].endswith("_B01_WTR.tif") and l["href"].startswith("https")]
        if m and url:
            k = (m.group(2), m.group(1))
            if k not in mejor or m.group(3) > mejor[k][0]: mejor[k] = (m.group(3), url[0])
    res = defaultdict(dict)
    for (f, t), (_, u) in mejor.items(): res[f][t] = u
    return dict(sorted(res.items()))

def leer_wtr(url):
    import earthaccess, rasterio
    from rasterio.io import MemoryFile
    for intento in range(1, 5):
        try:
            data = earthaccess.open([url])[0].read()
        except Exception as ex:
            p(f"    reintento {intento} ({type(ex).__name__}: {str(ex)[:70]})"); time.sleep(5); continue
        with MemoryFile(data) as mf, mf.open() as ds:
            ep = ds.crs.to_epsg()
            if ep // 100 not in (326, 327): raise ValueError(f"proyeccion DSWx inesperada {ds.crs}")
            # UTM 'norte' (326zz) -> mismo huso sur (327zz) sumando 10.000 km (ver src/agua_delta_dswx.py); devuelve el huso sur real
            return ds.read(1), ds.transform, (10_000_000.0 if ep // 100 == 326 else 0.0), 32700 + ep % 100
    return None, None, None, None

def a_celdas(a, tr, agua_vals, dys, grid, ep_src=None, ep_dst=None):
    ny, nx, bx0, by0, dx, dy = grid
    filas, cols = np.indices(a.shape)
    xs = tr.c + (cols + 0.5) * tr.a; ys = tr.f + (filas + 0.5) * tr.e + dys
    if ep_src and ep_dst and ep_src != ep_dst:   # p. ej. GCOV 111 D viene en UTM 20S y DSWx en 21S -> reproyectar
        from pyproj import Transformer
        xs, ys = Transformer.from_crs(f"EPSG:{ep_src}", f"EPSG:{ep_dst}", always_xy=True).transform(xs, ys)
    ci = np.floor((xs - bx0) / dx).astype(int); ri = np.floor((ys - by0) / dy).astype(int)
    ok = (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny)
    es_agua = np.isin(a, agua_vals); es_val = es_agua | (a == 0)
    idx = (ri * nx + ci)[ok]
    agua = np.bincount(idx, weights=es_agua[ok], minlength=ny * nx).reshape(ny, nx)
    val = np.bincount(idx, weights=es_val[ok], minlength=ny * nx).reshape(ny, nx)
    return agua, val

def correr_zona(zona, bbox, resumen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p(f"\n==================== ZONA {zona} bbox {bbox}")
    orbitas = gcov_por_orbita(bbox)
    for k, d in sorted(orbitas.items()):
        p(f"  GCOV {k[0]} {k[1]} frame {k[2]}: {sorted(d)}")
    for k, d in sorted(orbitas.items()):
        sel = elegir_fechas(list(d))
        tag = f"{k[0]}{k[1]}_{k[2]}"
        if sel is None:
            p(f"\n--- {tag}: no cumple la regla (>= 4 fechas y evento >= {EVENTO_MIN}) -> no se evalua"); continue
        p(f"\n--- ORBITA {tag}: ref {sel['ref']} | control {sel['control']} | evento {sel['evento']}")
        pil, fechas = [], []
        for f in sel["todas"]:
            r = leer_gcov(d[f], bbox)
            if r is None: p(f"  {f}: sin datos en la zona"); continue
            if pil and r["hv"].shape != pil[0]["hv"].shape: p(f"  {f}: grilla distinta, se omite"); continue
            pil.append(r); fechas.append(f)
            p(f"  {f}: {r['paso']:.0f} m -> {r['k']}x{r['k']} | HV mediana {10*np.log10(np.nanmedian(r['hv'])):.1f} dB")
        if any(x not in fechas for x in sel["ref"] + [sel["control"], sel["evento"]]):
            p("  falta alguna fecha clave -> NO EVALUABLE"); resumen[f"{zona}_{tag}"] = dict(estado="NO EVALUABLE (falta fecha)"); continue
        epsg = pil[0]["epsg"]; assert epsg // 100 == 327, f"se esperaba UTM sur (327zz), vino {epsg}"
        p(f"  GCOV en EPSG {epsg}")
        HV = np.stack([r["hv"] for r in pil]); HH = np.stack([r["hh"] for r in pil])
        # Calibracion relativa entre fechas = MISMO paso que la receta del Delta (src/agua_gcov_exploratorio.py calibrar(), notebook oficial ical=1).
        # Sin este paso, una fecha en que toda la escena se oscurece (humedad) marca miles de celdas falsas.
        fac = np.array([np.nanmean(HV[i]) for i in range(len(HV))]) / np.nanmean(HV)
        HV = HV / fac[:, None, None]; HH = HH / fac[:, None, None]
        p(f"  calibracion relativa (factor HV por fecha): {dict(zip(fechas, np.round(fac, 3).tolist()))}")
        ny, nx = HV.shape[1:]
        x0, y0, dx, dy = pil[0]["x0"], pil[0]["y0"], pil[0]["dx"], pil[0]["dy"]
        po = abs(dx) / pil[0]["k"]
        grid = (ny, nx, x0 - np.sign(dx) * po / 2, y0 - np.sign(dy) * po / 2, dx, dy)
        ref, d_ev, d_ct = cambios(HV, fechas, sel)

        # --- DSWx: ventanas de referencia, control y evento ---
        r0 = fd(sel["ref"][0]) - timedelta(days=VENT_REF); r1 = fd(sel["ref"][1]) + timedelta(days=VENT_REF)
        c0 = fd(sel["control"]) - timedelta(days=4); c1 = fd(sel["control"]) + timedelta(days=3)
        e0 = fd(sel["evento"]) + timedelta(days=VENT_EV[0]); e1 = fd(sel["evento"]) + timedelta(days=VENT_EV[1])
        en_ventana = lambda f: r0 <= fd(f) <= r1 or c0 <= fd(f) <= c1 or e0 <= fd(f) <= e1
        mapas = {}
        for prod, (cid, vals) in DSWX.items():
            F = dswx_por_fecha(cid, bbox, r0.strftime("%Y-%m-%dT00:00:00Z"), e1.strftime("%Y-%m-%dT23:59:59Z"))
            for f, tiles in F.items():
                if not en_ventana(f): continue
                ag = np.zeros((ny, nx)); va = np.zeros((ny, nx)); eps = set()
                for t, u in tiles.items():
                    a, tr, dys, ep = leer_wtr(u)
                    if a is None: continue
                    x, y = a_celdas(a, tr, vals, dys, grid, ep, epsg); ag += x; va += y; eps.add(ep)
                mapas[(prod, f)] = (ag, va)
                p(f"  DSWx-{prod} {f}: {len(tiles)} mosaicos, EPSG {sorted(eps)}, celdas con dato {np.mean(va > 0):.0%}")
        # agua previa (exclusion), con HLS + S1 juntos en la ventana de referencia
        agp = np.zeros((ny, nx)); vap = np.zeros((ny, nx))
        for (prod, f), (ag, va) in mapas.items():
            if r0 <= fd(f) <= r1: agp += ag; vap += va
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_previa = agp / vap
        excluir = ~(vap > 0) | (frac_previa >= AGUA_PREVIA)
        M, C, Mc = grupos(d_ev, d_ct, excluir)
        p(f"  celdas: zona {np.isfinite(ref).sum()} | excluidas (agua previa o sin DSWx) {int((excluir & np.isfinite(ref)).sum())} | "
          f"MARCADO {M.sum()} | CONTROL {C.sum()} | marcadas con fecha control {Mc.sum()}")

        verdad, series = {}, {}
        for prod in DSWX:
            serie = []
            for (pp_, f), (ag, va) in sorted(mapas.items()):
                if pp_ != prod: continue
                cub = va > 0
                validos = float(cub[np.isfinite(ref)].mean())
                pct = lambda X: float(100 * ag[X & cub].sum() / max(va[X & cub].sum(), 1))
                serie.append(dict(fecha=f, validos=validos, M=pct(M), C=pct(C)))
            series[prod] = serie
            p(f"  DSWx-{prod}: fecha  validos  %agua MARCADO  %agua CONTROL")
            for s in serie:
                p(f"     {s['fecha']}  {s['validos']:5.0%}   {s['M']:8.1f}       {s['C']:8.1f}" + ("" if s["validos"] >= MIN_VALIDO else "  (nubes/sin dato: no se usa)"))
            verdad[prod] = resumen_dswx(serie, sel)
            p(f"  DSWx-{prod} resumen: {verdad[prod]}")
        r = evaluar(M.sum(), Mc.sum(), verdad)
        r.update(orbita=tag, fechas=sel, verdad=verdad)
        p(f"  >>> A1 {r['A1']} | A2 {r['A2']} | A3 {r['A3']} ({r['nMc']} vs {r['nM']}) | A4 {r['A4']} | producto {r['prod']} => {r['estado']}")
        resumen[f"{zona}_{tag}"] = r

        np.savez_compressed(os.path.join(OUT, f"{zona}_{tag}.npz"), HV=HV.astype(np.float32), HH=HH.astype(np.float32),
                            fechas=np.array(fechas), d_ev=d_ev.astype(np.float32), marcado=M, control=C, excluir=excluir,
                            x0=x0, y0=y0, dx=dx, dy=dy, epsg=epsg)
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        ax[0, 0].imshow(10 * np.log10(ref), cmap="gray", vmin=-30, vmax=-10, interpolation="nearest"); ax[0, 0].set_title(f"HV referencia (dB) {sel['ref']}")
        ax[0, 1].imshow(d_ev, cmap="RdBu", vmin=-8, vmax=8, interpolation="nearest"); ax[0, 1].set_title(f"cambio HV evento {sel['evento']} (dB)")
        ax[0, 2].imshow(d_ct, cmap="RdBu", vmin=-8, vmax=8, interpolation="nearest"); ax[0, 2].set_title(f"cambio HV CONTROL {sel['control']} (dB)")
        cod = np.where(excluir, 1, 0) + np.where(M, 2, 0) + np.where(C, 3, 0)
        cm = matplotlib.colors.ListedColormap(["white", "lightsteelblue", "crimson", "lightgray"])
        ax[1, 0].imshow(cod, cmap=cm, vmin=0, vmax=3, interpolation="nearest"); ax[1, 0].set_title("rojo MARCADO | gris CONTROL | celeste agua previa/excluida")
        for j, prod in enumerate(DSWX):
            ss = [s for s in series[prod] if s["validos"] >= MIN_VALIDO]
            xs = [datetime.strptime(s["fecha"], "%Y%m%d") for s in ss]
            ax[1, 1 + j].plot(xs, [s["M"] for s in ss], "o-", color="crimson", label="MARCADO")
            ax[1, 1 + j].plot(xs, [s["C"] for s in ss], "s-", color="gray", label="CONTROL")
            ax[1, 1 + j].axvline(fd(sel["evento"]), color="k", ls="--", alpha=.5)
            ax[1, 1 + j].set_title(f"OPERA DSWx-{prod}: % agua"); ax[1, 1 + j].legend(); ax[1, 1 + j].tick_params(axis="x", rotation=45)
        fig.suptitle(f"Crecida Parana/Paraguay - {zona} - GCOV {tag} - {r['estado']}", fontsize=13)
        plt.tight_layout(); fig.savefig(os.path.join(OUT, f"{zona}_{tag}.png"), dpi=85); plt.close(fig)
        p(f"  figura: {os.path.join(OUT, zona + '_' + tag + '.png')}")

if __name__ == "__main__":
    if "--prueba" in sys.argv:
        prueba(); sys.exit(0)
    print("Cargando librerias (1-3 min en Windows, no cortar)...", flush=True)
    import earthaccess, h5py, rasterio, requests, pyproj, matplotlib
    from rasterio.io import MemoryFile
    print("Librerias cargadas.", flush=True)
    earthaccess.login(persist=True)
    LOG = open(os.path.join(OUT, "informe.txt"), "w", encoding="utf-8")
    pedidas = [a for a in sys.argv[1:] if a in ZONAS] or list(ZONAS)
    resumen = {}
    for z in pedidas:
        correr_zona(z, ZONAS[z], resumen)
    p("\n==================== RESUMEN")
    for k, r in resumen.items():
        p(f"  {k}: {r.get('estado')}")
    json.dump(resumen, open(os.path.join(OUT, "resumen.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    LOG.close()
