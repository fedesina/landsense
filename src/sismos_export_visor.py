"""
sismos_export_visor.py -- LandSense
Arma web/sismos.json para el globo del visor: TODOS los sismos M>=6 (prof<=40 km) desde el 17-jun-2026
con su estado frente a NISAR. No inventa nada: une
  - data/processed/sismos/catalogo.json  (src/sismos_catalogo.py: coordenadas USGS, magnitud, profundidad)
  - data/processed/sismos/tabla_final.md (estado por sismo, revisado a mano)
Si un sismo del catalogo no aparece en la tabla, el script se detiene (no se adivina el estado).

Semaforo del visor:
  verde    = VISTO Y VERIFICADO (o VALIDADO)
  amarillo = indicio: VISTO, COINCIDE CON USGS / VISTO, SIN VERIFICAR
  azul     = TODAVIA SIN DATO (se espera una pasada)
  gris     = NO SE PUEDE (mar o profundo)

Uso:  python src/sismos_export_visor.py
Re-correr cada vez que cambie tabla_final.md.
"""
import os, json, re

AQUI = os.path.dirname(os.path.abspath(__file__))
SIS = os.path.join(AQUI, "..", "data", "processed", "sismos")
OUT = os.path.join(AQUI, "..", "web", "sismos.json")

SEMAFORO = [
    ("VISTO Y VERIFICADO", "verde", "Visto y verificado"),
    ("VISTO, COINCIDE CON USGS", "amarillo", "Visto: coincide con USGS"),
    ("VISTO, SIN VERIFICAR", "amarillo", "Visto, falta confirmar"),
    ("TODAVIA SIN DATO", "azul", "Esperando la próxima pasada"),
    ("NO SE PUEDE", "gris", "NISAR no puede medirlo"),
]
EXPLICA = {
    "verde": "NISAR midió cuánto se movió el suelo y una fuente independiente (NASA) lo confirma.",
    "amarillo": "NISAR vio movimiento del suelo; todavía falta una confirmación independiente completa.",
    "azul": "El sismo es reciente: el satélite todavía no pasó con la geometría necesaria o el dato no se publicó.",
    "gris": "El sismo fue bajo el mar o muy profundo: el radar mide la superficie de la tierra firme, no el fondo del mar.",
}
# Por CODIGO USGS (USGS cambio el texto del lugar de varios sismos y por nombre se perdia la escena)
ESCENA = {"us6000t7zp": "la_guaira", "us6000u18k": "panama", "us6000u43g": "tibet"}
# Texto para el publico (lenguaje simple) de los sismos que NISAR vio. Resume los resultados revisados; no agrega datos.
SIMPLE = {
    "us6000t7zp":
        "NISAR midió que el suelo se corrió unos 56 cm hacia el oeste al sur del aeropuerto de Maiquetía. NASA/JPL midió lo mismo (hasta unos 60 cm).",
    "us6000t7zc":
        "Fue parte del mismo doble sismo que La Guaira. NISAR ve una falla que corta el suelo de este a oeste, con los dos lados moviéndose en sentido contrario, igual que lo que indica el análisis sísmico de USGS. Es un indicio: esa comparación la definimos después de ver el mapa.",
    "us6000u18k":
        "Dos órbitas de NISAR ven el suelo romperse a lo largo de la falla, con el lado norte corrido hacia el oeste: igual que lo que indica el análisis sísmico de USGS. No damos los centímetros hacia el oeste o hacia arriba: la otra geometría del satélite salió con errores de procesamiento.",
    "us6000u19w":
        "Ocurrió la misma tarde y cerca del sismo de magnitud 7,7: el radar ve la suma de los dos movimientos y no puede separarlos.",
    "us6000tgb9":
        "Cerca de la falla el suelo se movió metros, y con las pasadas disponibles la medición fina se rompe. El sistema ve el sismo pero se niega a darlo por verificado.",
    "us6000u43g":
        "Dos órbitas de NISAR ven el suelo moverse decenas de centímetros alrededor del epicentro, y coinciden. Falta una pasada desde el otro lado del satélite para saber cuánto fue hundimiento y cuánto movimiento de costado.",
}


def leer_tabla(path):
    filas = {}
    for linea in open(path, encoding="utf-8"):
        if not linea.startswith("| 2026-"):
            continue
        c = [x.strip() for x in linea.strip().strip("|").split("|")]
        fecha, mag, prof, lugar, resultado, detalle = c[:6]
        filas[(fecha, lugar)] = dict(resultado=resultado, detalle=detalle)
    return filas


def main():
    cat = json.load(open(os.path.join(SIS, "catalogo.json"), encoding="utf-8"))
    tabla = leer_tabla(os.path.join(SIS, "tabla_final.md"))
    out, faltan = [], []
    for s in cat:
        t = tabla.get((s["fecha"], s["lugar"]))
        if t is None:
            faltan.append(f'{s["fecha"]} {s["lugar"]}'); continue
        sem = next((x for x in SEMAFORO if t["resultado"].startswith(x[0])), None)
        if sem is None:
            raise SystemExit(f"estado desconocido en tabla_final.md: {t['resultado']}")
        out.append(dict(id=s["id"], fecha=s["fecha"], mag=s["mag"], prof_km=s["prof_km"], lugar=s["lugar"],
                        lon=s["lon"], lat=s["lat"], color=sem[1], estado=sem[2], estado_tabla=t["resultado"],
                        detalle=t["detalle"], explica=SIMPLE.get(s["id"], EXPLICA[sem[1]]), escena=ESCENA.get(s["id"])))
    if faltan:
        raise SystemExit("Sismos del catalogo sin estado en tabla_final.md (no se adivina): " + "; ".join(faltan))
    res = {k: sum(1 for x in out if x["color"] == k) for k in ("verde", "amarillo", "azul", "gris")}
    en_tierra = res["verde"] + res["amarillo"]
    doc = dict(_nota="Generado por src/sismos_export_visor.py desde catalogo.json (USGS) + tabla_final.md. No editar a mano.",
               periodo=f"17-jun-2026 a {max(x['fecha'] for x in out)}", criterio="USGS M>=6,0, profundidad <= 40 km",
               resumen=dict(total=len(out), **res, vistos=en_tierra), sismos=out)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"web/sismos.json: {len(out)} sismos | {res}")


if __name__ == "__main__":
    main()
