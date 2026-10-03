"""Regresiones del verificador: no aceptar resultados vacíos ni cambios científicos ocultos."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from comparar_resultados import comparar_csv, comparar_json
from excepciones_comparacion import comprobar_excepcion
from test_basico import correr


class TestValores(unittest.TestCase):
    def test_nan_unilateral_falla_en_ambos_sentidos(self):
        for a, b in [(float('nan'), 1.0), (1.0, float('nan'))]:
            self.assertTrue(comparar_json({'x': a}, {'x': b}, 1e-9, 1e-12)['fallas'])

    def test_nan_bilateral_coincide(self):
        self.assertFalse(comparar_json(float('nan'), float('nan'), 1e-9, 1e-12)['fallas'])

    def test_bool_no_es_numero(self):
        self.assertTrue(comparar_json(True, 1, 1e-9, 1e-12)['fallas'])

    def test_encabezado_es_texto_exacto(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / 'a.csv', Path(t) / 'b.csv'
            a.write_text('2012\n1\n'); b.write_text('2012.0\n1\n')
            self.assertTrue(comparar_csv(a, b, 1e-9, 1e-12)['fallas'])


class TestExcepciones(unittest.TestCase):
    def test_cambio_bibliografico_no_oculta_cifras(self):
        with tempfile.TemporaryDirectory() as t:
            ref, run = Path(t) / 'ref.csv', Path(t) / 'run.csv'
            ref.write_text('fuente,valor\nZewde (ref. 23),1.93\n')
            regla = {'tipo': 'csv_sustitucion_texto', 'origen': 'Zewde (ref. 23)', 'destino': 'Zewde (ref. 26)'}
            run.write_text('fuente,valor\nZewde (ref. 26),1.93\n')
            self.assertTrue(comprobar_excepcion({'estado': 'difiere'}, regla, ref, run, 1e-9, 1e-12))
            run.write_text('fuente,valor\nZewde (ref. 26),2.93\n')
            self.assertFalse(comprobar_excepcion({'estado': 'difiere'}, regla, ref, run, 1e-9, 1e-12))

    def test_omision_json_no_oculta_cambio_en_otras_claves(self):
        with tempfile.TemporaryDirectory() as t:
            ref, run = Path(t) / 'ref.json', Path(t) / 'run.json'
            ref.write_text(json.dumps({'n': 82859, 'cotejo': {'valor': 1}}))
            regla = {'tipo': 'json_claves_opcionales', 'claves': ['cotejo']}
            run.write_text(json.dumps({'n': 82859}))
            self.assertTrue(comprobar_excepcion({'estado': 'difiere'}, regla, ref, run, 1e-9, 1e-12))
            run.write_text(json.dumps({'n': 82858}))
            self.assertFalse(comprobar_excepcion({'estado': 'difiere'}, regla, ref, run, 1e-9, 1e-12))
            run.write_text(json.dumps({'n': 82859, 'cotejo': {'valor': 2}}))
            self.assertFalse(comprobar_excepcion({'estado': 'difiere'}, regla, ref, run, 1e-9, 1e-12))

    def test_salida_nueva_solo_acepta_hash_esperado(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t) / 'run.csv'; run.write_text('x\n1\n')
            regla = {'tipo': 'salida_nueva_sha256', 'sha256': hashlib.sha256(run.read_bytes()).hexdigest()}
            self.assertTrue(comprobar_excepcion({'estado': 'sin_referencia'}, regla, Path(t) / 'ref.csv', run, 1e-9, 1e-12))
            run.write_text('x\n2\n')
            self.assertFalse(comprobar_excepcion({'estado': 'sin_referencia'}, regla, Path(t) / 'ref.csv', run, 1e-9, 1e-12))

    def test_ausencia_no_exime_una_diferencia(self):
        self.assertFalse(comprobar_excepcion({'estado': 'difiere'}, {'tipo': 'ausencia_documentada'}, Path('ref'), Path('run'), 1e-9, 1e-12))


class TestComparacionCLI(unittest.TestCase):
    def test_carpeta_vacia_rechazada(self):
        with tempfile.TemporaryDirectory() as t:
            r = correr('comparar', '--output-dir', t)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertFalse((Path(t) / 'comparacion_referencia.json').exists())

    def test_figuras_incompletas_fallan(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / 'figuras').mkdir()
            r = correr('comparar', '--output-dir', t)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            j = json.loads((Path(t) / 'comparacion_referencia.json').read_text())
            self.assertEqual(len(j['inventario_figuras']['faltantes']), 12)

    def test_tolerancia_nan_rechazada(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / 'benchmark').mkdir()
            r = correr('comparar', '--output-dir', t, '--rtol', 'nan')
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)


if __name__ == '__main__':
    unittest.main()
