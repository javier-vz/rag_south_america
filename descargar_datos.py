#!/usr/bin/env python3
"""Descarga SOLO LECTURA de tres archivos públicos del repositorio original.

Verifica que los bytes sean idénticos a los blobs de main auditados el 09-11-2026.
Nunca usa credenciales, nunca escribe en GitHub ni sobrescribe archivos locales.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://raw.githubusercontent.com/javier-vz/RAG_SA/main/"
EXPECTED_GIT_BLOBS = {
    "grafo_enriquecido.ttl": "f39902dcae4356e3a592d979d2f4f6b7f35c46bf",
    "embed_matrix_v2.npy": "99ffd368b5ee9b42aa01b6111af5ffae4726c5e6",
    "id_map_v2.pkl": "075beac96736142dacb6ed7dea180fc506cab81e",
}
ROOT = Path(__file__).resolve().parent


def git_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def main():
    for filename, expected in EXPECTED_GIT_BLOBS.items():
        target = ROOT / filename
        if target.exists():
            print(f"[YA EXISTE] {filename} — no se sobrescribe")
            continue
        request = Request(BASE + filename, headers={"User-Agent": "RAG-SA-data-setup/1.0"})
        print(f"Descargando (solo lectura): {filename}")
        with urlopen(request, timeout=35) as response:
            content = response.read(10_000_001)
        if len(content) > 10_000_000:
            raise RuntimeError(f"Archivo inesperadamente grande: {filename}")
        if git_sha1(content) != expected:
            raise RuntimeError(
                f"El contenido de {filename} cambió respecto de la versión auditada. "
                "No se guardó; copia manualmente tus archivos originales si son una versión más reciente."
            )
        target.write_bytes(content)
        print(f"[OK] {filename} ({len(content):,} bytes; hash verificado)")
    print("Datos preparados. Ejecuta: streamlit run app_lenguas.py")


if __name__ == "__main__":
    main()
