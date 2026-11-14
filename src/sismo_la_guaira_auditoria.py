"""
sismo_la_guaira_auditoria.py -- LandSense
AUDITORIA del co-sismico La Guaira (24-jun-2026) antes de llevarlo al 3D.

Que verifica (sin suponer nada):
  1. Procesa TODOS los pares NISAR GUNW que cruzan el sismo (NASA/JPL uso
     13->25 jun y 18->30 jun). Si la senal aparece en ambos pares, en el mismo
     lugar, es real; si aparece en uno solo, sospechar.
  2. Componentes conectados: el desenrollado de fase puede dejar zonas corridas
     entre si por multiplos de lambda/2 (~11.9 cm). Mide el salto en cada borde
     entre componentes y lo expresa en "ciclos". Salto ~entero = sospecha.
  3. Imprime los atributos del producto (unidades, descripcion, direccion de
     orbita/mirada) para fijar la CONVENCION DE SIGNO antes de pintar nada.
  4. Guarda el dato georreferenciado (npz + json) para el visor 3D.

Uso:
    python src/sismo_la_guaira_auditoria.py
Salidas: ../data/processed/la_guaira/  (npz por par, auditoria.png, auditoria.txt)
"""
print("Iniciando auditoria La Guaira... (cargar librerias puede tardar 1-2 min la primera vez)", flush=True)
import os, re, time, json
import numpy as np
import h5py
import earthaccess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
print("Librerias cargadas. Login Earthdata...", flush=True)

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "la_guaira")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "auditoria.txt"), "w", encoding="utf-8")

def p(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

LON_MIN, LON_MAX = -67.60, -66.50
LAT_MIN, LAT_MAX = 10.30, 10.80
SISMO = "20260624"
LAMBDA_L = 0.238                       # m (aproximado; los demas scripts leen lambda del producto: 0.2420 m)
CICLO_CM = LAMBDA_L / 2 * 100          # un ciclo de fase (2pi) en LOS = lambda/2
BASE = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
AEROPUERTO = (-66.99, 10.60)           # Maiquetia aprox (referencia de NASA: zona fuerte al sur)

earthaccess.login(persist=True)
res = earthaccess.search_data(short_name="NISAR_L2_GUNW_PROVISIONAL_V1",
                              bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
                              temporal=("2026-06-10", "2026-07-15"))
p(f"GUNW encontrados en la ventana: {len(res)}")

pares = []
for g in res:
    links = g.data_links()
    name = links[0].split("/")[-1] if links else str(g)
    d = re.findall(r"(\d{8})T\d{6}", name)
    if len(d) >= 3:
        p(f"  {name}")
        if d[0] <= SISMO <= d[2]:
            pares.append((g, name, d[0], d[2]))
p(f"\nPares que cruzan el sismo: {len(pares)}")
for _, n, r, s in pares:
    p(f"  {r} -> {s}   {n}")
if not pares:
    raise SystemExit("Ningun par cruza el sismo en el AOI.")

def attrs(f, path):
    try:
        return {k: (v.decode() if isinstance(v, bytes) else
                    (v.tolist() if hasattr(v, "tolist") else v)) for k, v in f[path].attrs.items()}
    except Exception as e:
        return f"(sin attrs: {e})"

def escalar(f, path):
    try:
        v = f[path][()]
        return v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)
    except Exception:
        return "(no existe)"

def leer(fobj):
    with h5py.File(fobj, "r") as f:
        meta = {
            "orbitPassDirection": escalar(f, "science/LSAR/identification/orbitPassDirection"),
            "lookDirection": escalar(f, "science/LSAR/identification/lookDirection"),
            "attrs_unwrappedPhase": attrs(f, f"{BASE}/unwrappedPhase"),
            "attrs_ionosphere": attrs(f, f"{BASE}/ionospherePhaseScreen"),
            "attrs_connectedComponents": attrs(f, f"{BASE}/connectedComponents"),
            "centerFrequency_Hz": escalar(f, "science/LSAR/GUNW/grids/frequencyA/centerFrequency"),
        }
        x = f[f"{BASE}/xCoordinates"][:]; y = f[f"{BASE}/yCoordinates"][:]
        epsg = int(f[f"{BASE}/projection"][()])
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xa, ya = tr.transform(LON_MIN, LAT_MIN); xb, yb = tr.transform(LON_MAX, LAT_MAX)
        c = np.where((x >= min(xa, xb)) & (x <= max(xa, xb)))[0]
        r = np.where((y >= min(ya, yb)) & (y <= max(ya, yb)))[0]
        r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
        d = dict(
            ph=f[f"{BASE}/unwrappedPhase"][r0:r1, c0:c1].astype(np.float32),
            io=f[f"{BASE}/ionospherePhaseScreen"][r0:r1, c0:c1].astype(np.float32),
            co=f[f"{BASE}/coherenceMagnitude"][r0:r1, c0:c1].astype(np.float32),
            cc=f[f"{BASE}/connectedComponents"][r0:r1, c0:c1].astype(np.int32),
            x=x[c0:c1], y=y[r0:r1], epsg=epsg)
    return d, meta

def saltos_entre_componentes(los_cm, cc):
    """Mediana de la diferencia LOS a traves de cada borde entre componentes distintos."""
    res = {}
    for a1, a2, b1, b2 in [(los_cm[:, :-1], los_cm[:, 1:], cc[:, :-1], cc[:, 1:]),
                           (los_cm[:-1, :], los_cm[1:, :], cc[:-1, :], cc[1:, :])]:
        m = (b1 != b2) & (b1 > 0) & (b2 > 0) & np.isfinite(a1) & np.isfinite(a2)
        for u, v, d in zip(b1[m], b2[m], (a2 - a1)[m]):
            key = (min(u, v), max(u, v))
            res.setdefault(key, []).append(d if u < v else -d)
    return {k: (np.median(v), len(v)) for k, v in res.items()}

resultados = []
for g, name, ref, sec in pares:
    p(f"\n========== PAR {ref} -> {sec} ==========")
    for intento in range(1, 5):
        try:
            d, meta = leer(earthaccess.open([g])[0]); break
        except Exception as e:
            p(f"  reintento {intento} ({type(e).__name__}: {str(e)[:80]})"); time.sleep(6)
    else:
        p("  No se pudo leer este par."); continue

    p("Metadatos (para fijar convencion de signo):")
    for k, v in meta.items():
        p(f"  {k}: {v}")

    los_cm = (d["ph"] - np.nan_to_num(d["io"])) * LAMBDA_L / (4 * np.pi) * 100
    valido = np.isfinite(los_cm) & (d["co"] >= 0.2) & (d["cc"] > 0)
    los_cm = np.where(valido, los_cm, np.nan)

    comps = [(int(k), int(n)) for k, n in zip(*np.unique(d["cc"][valido], return_counts=True))]
    comps.sort(key=lambda t: -t[1])
    p(f"Pixeles validos: {int(valido.sum())} | componentes conectados: {len(comps)}")
    for k, n in comps[:10]:
        v = los_cm[d["cc"] == k]
        p(f"  comp {k:3d}: {n:8d} px | LOS mediana {np.nanmedian(v):+7.1f} cm | "
          f"p5 {np.nanpercentile(v,5):+7.1f} p95 {np.nanpercentile(v,95):+7.1f}")

    p("Saltos en bordes entre componentes (>=200 px de borde):")
    for (a, b), (med, n) in sorted(saltos_entre_componentes(los_cm, d["cc"]).items(), key=lambda t: -t[1][1]):
        if n < 200: continue
        cic = med / CICLO_CM
        p(f"  comp {a}|{b}: salto {med:+7.1f} cm = {cic:+5.2f} ciclos (borde {n} px)"
          f"{'  <-- ~entero: SOSPECHA de ambiguedad de desenrollado' if abs(cic) >= 0.7 and abs(cic - round(cic)) < 0.2 else ''}")

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{d['epsg']}", always_xy=True)
    ax_, ay_ = tr.transform(*AEROPUERTO)
    ix = int(np.abs(d["x"] - ax_).argmin()); iy = int(np.abs(d["y"] - ay_).argmin())
    ven = los_cm[max(0, iy - 60):iy + 1, max(0, ix - 30):ix + 30]   # ~5 km al sur del aeropuerto
    p(f"Zona al sur del aeropuerto (referencia NASA ~60 cm): mediana {np.nanmedian(ven):+.1f} cm, "
      f"validos {np.isfinite(ven).sum()}")

    np.savez_compressed(os.path.join(OUT, f"gunw_{ref}_{sec}.npz"), **d)
    json.dump({"ref": ref, "sec": sec, "archivo": name, "epsg": d["epsg"],
               "lambda_m_usada": LAMBDA_L, "meta": meta},
              open(os.path.join(OUT, f"gunw_{ref}_{sec}.json"), "w", encoding="utf-8"),
              indent=1, default=str)
    resultados.append((ref, sec, los_cm, d, (ax_, ay_)))

# Figura comparativa: mismos colores para todos los pares
n = len(resultados)
fig, ax = plt.subplots(2, n, figsize=(7 * n, 10), squeeze=False)
for j, (ref, sec, los_cm, d, (axx, ayy)) in enumerate(resultados):
    ext = [d["x"][0], d["x"][-1], d["y"][-1], d["y"][0]]
    im = ax[0, j].imshow(los_cm, cmap="RdBu", vmin=-70, vmax=70, extent=ext, origin="upper")
    ax[0, j].plot(axx, ayy, "k*", ms=14); ax[0, j].set_title(f"LOS (cm) {ref}->{sec}\n(* = aeropuerto)")
    plt.colorbar(im, ax=ax[0, j])
    ccv = np.where(d["cc"] > 0, d["cc"], np.nan).astype(float)
    im2 = ax[1, j].imshow(ccv, cmap="tab20", extent=ext, origin="upper", interpolation="nearest")
    ax[1, j].set_title("Componentes conectados"); plt.colorbar(im2, ax=ax[1, j])
plt.tight_layout()
fig.savefig(os.path.join(OUT, "auditoria.png"), dpi=110)
p(f"\nFigura: {os.path.join(OUT, 'auditoria.png')}")
p("Listo. Pegar auditoria.txt completo y mirar auditoria.png.")
LOG.close()
