"""Escritura de web/puntos/<id>/punto.json para el visor.

Cada punto.json guarda sus textos en ingles en la clave "_en" (ruta del texto -> traduccion), que el visor
aplica con ?lang=en. Los scripts de export arman el punto.json desde cero, asi que al volver a correrlos
esa clave se perderia: guardar_punto() la conserva si el archivo ya existia.
"""
import json, os


def guardar_punto(ruta, punto):
    if "_en" not in punto and os.path.exists(ruta):
        try:
            with open(ruta, encoding="utf-8") as f:
                viejo = json.load(f)
        except (OSError, ValueError):
            viejo = {}
        if "_en" in viejo:
            punto["_en"] = viejo["_en"]
            print(f"  textos en ingles conservados en {ruta} (revisarlos si cambio el texto en castellano)")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(punto, f, ensure_ascii=False, indent=1)
