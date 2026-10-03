"""Excepciones acotadas: una diferencia conocida nunca exime al archivo completo."""
import json
import tempfile
from pathlib import Path
from comparar_resultados import comparar_csv, comparar_json, sha


def comprobar_excepcion(fila, regla, referencia, ejecucion, rtol, atol):
    """Devuelve True solo si se verifica exactamente la excepción documentada."""
    estado, tipo = fila['estado'], regla['tipo']
    if tipo == 'ausencia_documentada':
        return estado == 'no_generado'
    if tipo == 'salida_nueva_sha256':
        return estado == 'sin_referencia' and ejecucion.is_file() and sha(ejecucion) == regla['sha256']
    if estado != 'difiere':
        return False
    if tipo == 'json_claves_opcionales':
        ref = json.loads(referencia.read_text(encoding='utf-8'))
        run = json.loads(ejecucion.read_text(encoding='utf-8'))
        # Solo se permite la omisión, no el cambio, de estas claves de cotejo histórico.
        if not isinstance(ref, dict) or not isinstance(run, dict):
            return False
        for clave in regla['claves']:
            if clave not in run:
                ref.pop(clave, None)
        return not comparar_json(run, ref, rtol, atol)['fallas']
    if tipo == 'csv_sustitucion_texto':
        texto = referencia.read_text(encoding='utf-8')
        origen, destino = regla['origen'], regla['destino']
        if texto.count(origen) != 1:
            return False
        with tempfile.TemporaryDirectory() as temporal:
            normalizada = Path(temporal) / 'referencia.csv'
            normalizada.write_text(texto.replace(origen, destino), encoding='utf-8')
            return not comparar_csv(ejecucion, normalizada, rtol, atol)['fallas']
    return False
