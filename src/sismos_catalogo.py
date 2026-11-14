"""
sismos_catalogo.py -- LandSense
TABLA GLOBAL: todos los sismos fuertes desde que NISAR tiene datos PROVISIONAL, y si NISAR los puede ver.

CRITERIOS FIJADOS ANTES de mirar:
  Sismos: catálogo USGS, M >= 6.0, profundidad <= 40 km, desde 2026-06-17 (inicio PROVISIONAL) hasta hoy.
  "Par que cruza" = GUNW PROVISIONAL con fecha de referencia ANTES del sismo y secundaria DESPUÉS.
  Un par "cubre" el sismo si su polígono contiene el epicentro o pasa a < 30 km de él.
  Clase NISAR:
     A+D  = hay par que cubre en órbita ascendente Y en descendente (se puede intentar la descomposición)
     1 DIR = solo una dirección
     SIN PAR CERCA = ningún par GUNW a < 30 km (típico: sismo en el mar; o hueco de datos)
  Estado del análisis: se lee de data/processed/sismos/<id>/resumen.json (src/sismo_generico.py) si existe;
  si no, "no analizado". El estado humano (VERIFICADO/VALIDADO) se carga a mano en tabla_final.md despues de revisar cada caso; no se inventa acá.
  Esta tabla NO dice que NISAR haya medido nada: solo qué sismos tienen datos para intentarlo.

Uso (no necesita login: USGS y CMR son públicos):
    python src/sismos_catalogo.py
Salidas: data/processed/sismos/catalogo.csv, catalogo.txt, catalogo.json
"""
import os, re, json, csv, math
from datetime import datetime, timezone
import requests
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "data", "processed", "sismos")
os.makedirs(OUT, exist_ok=True)
M_MIN, PROF_MAX, INICIO, DIST_MAX = 6.0, 40.0, "2026-06-17", 30.0
COL_GUNW = "C2854335566-ASF"

def dist_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def dentro(poly, lon, lat):
    from matplotlib.path import Path
    return bool(Path(poly).contains_point((lon, lat)))

def dist_poligono(poly, lon, lat):
    """0 si el punto está dentro; si no, distancia (km) al borde (densificado cada ~1 km)."""
    if dentro(poly, lon, lat):
        return 0.0
    d = 1e9
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        n = max(2, int(dist_km(x0, y0, x1, y1)))
        for t in np.linspace(0, 1, n):
            d = min(d, dist_km(x0 + t * (x1 - x0), y0 + t * (y1 - y0), lon, lat))
    return d

def prueba():
    cuad = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert dist_poligono(cuad, 0.5, 0.5) == 0.0
    d = dist_poligono(cuad, 1.2, 0.5)                    # 0,2° al este del borde en el ecuador ≈ 22 km
    assert 20 < d < 24, d
    print("Prueba del código con archivos de prueba: OK (geometría inventada, no son datos reales).")

def main():
    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    q = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=90, params=dict(
        format="geojson", starttime=INICIO, endtime=hoy + "T23:59:59", minmagnitude=M_MIN, maxdepth=PROF_MAX, orderby="time")).json()
    sismos = q["features"]
    filas = []
    for f in sismos:
        pr = f["properties"]; lon, lat, prof = f["geometry"]["coordinates"]
        t = datetime.fromtimestamp(pr["time"] / 1000, timezone.utc)
        tstr = t.strftime("%Y%m%dT%H%M%S")
        d = 0.5                                           # caja de búsqueda ~55 km
        E = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", timeout=90, params=dict(
            collection_concept_id=COL_GUNW, bounding_box=f"{lon-d},{lat-d},{lon+d},{lat+d}",
            temporal=f"{INICIO}T00:00:00Z,{hoy}T23:59:59Z",
            page_size=500)).json()["feed"]["entry"]
        pares = {}
        for e in E:
            m = re.search(r"GUNW_\d{3}_(\d{3})_([AD])_(\d{3})_\d{3}_\d{4}_\w{2}_(\d{8}T\d{6})_\d{8}T\d{6}_(\d{8}T\d{6})_", e["title"])
            if not m: continue
            trk, dr, frm, ref, sec = m.groups()
            if not (ref < tstr < sec): continue
            polys = e.get("polygons") or []
            if not polys: continue
            v = list(map(float, polys[0][0].split()))
            poly = list(zip(v[1::2], v[0::2]))
            dk = dist_poligono(poly, lon, lat)
            dias = (datetime.strptime(sec[:8], "%Y%m%d") - datetime.strptime(ref[:8], "%Y%m%d")).days
            k = (trk, dr)
            if k not in pares or (dk, dias) < (pares[k]["dist"], pares[k]["dias"]):
                pares[k] = dict(track=trk, dir=dr, frame=frm, ref=ref[:8], sec=sec[:8], dias=dias, dist=round(dk, 1))
        cerca = [p for p in pares.values() if p["dist"] < DIST_MAX]
        dirs = {p["dir"] for p in cerca}
        clase = "A+D" if dirs == {"A", "D"} else ("1 DIR" if dirs else "SIN PAR CERCA")
        rj = os.path.join(OUT, f["id"], "resumen.json")
        estado = json.load(open(rj, encoding="utf-8")).get("estado", "?") if os.path.exists(rj) else "no analizado"
        filas.append(dict(id=f["id"], fecha=t.strftime("%Y-%m-%d"), mag=pr["mag"], prof_km=round(prof, 1), lugar=pr["place"],
                          lon=round(lon, 3), lat=round(lat, 3), clase_nisar=clase,
                          pares=" ; ".join(f"{p['track']}{p['dir']} {p['ref']}->{p['sec']} ({p['dias']} d, {p['dist']} km)" for p in sorted(cerca, key=lambda p: p["dist"])),
                          dias_min=min([p["dias"] for p in cerca], default=None), estado_src33=estado))
    with open(os.path.join(OUT, "catalogo.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    json.dump(filas, open(os.path.join(OUT, "catalogo.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    lines = [f"Sismos USGS M>={M_MIN}, prof<={PROF_MAX} km, {INICIO} -> {hoy}: {len(filas)}",
             f"Clase NISAR: A+D {sum(r['clase_nisar']=='A+D' for r in filas)} | 1 DIR {sum(r['clase_nisar']=='1 DIR' for r in filas)} | "
             f"SIN PAR CERCA {sum(r['clase_nisar']=='SIN PAR CERCA' for r in filas)}", ""]
    for r in filas:
        lines.append(f"{r['fecha']} M{r['mag']:.1f} {r['prof_km']:>5} km  {r['clase_nisar']:<13} {r['lugar'][:45]:<45} [{r['id']}] {r['estado_src33']}")
        if r["pares"]: lines.append(f"      {r['pares']}")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(OUT, "catalogo.txt"), "w", encoding="utf-8").write(txt + "\n")

if __name__ == "__main__":
    import sys
    prueba() if len(sys.argv) > 1 and sys.argv[1] == "prueba" else main()
