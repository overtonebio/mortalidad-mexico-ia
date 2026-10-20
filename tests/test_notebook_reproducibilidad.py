"""Seguridad y fidelidad de las adaptaciones del notebook; sin datos externos."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('soporte_notebook', REPO / 'notebooks/soporte_reproducibilidad.py')
soporte = importlib.util.module_from_spec(SPEC)
try:
    import pandas as pd
    SPEC.loader.exec_module(soporte)
except ModuleNotFoundError as error:
    if error.name in {'numpy', 'pandas', 'IPython'}:
        raise unittest.SkipTest('Estas pruebas requieren environments/notebook.txt') from error
    raise


class NotebookReproducibilidadTests(unittest.TestCase):
    def test_cada_ejecucion_es_nueva_y_no_pisa_entradas(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            datos = repo / 'data/external'
            datos.mkdir(parents=True)
            (datos / 'original').write_text('sin cambios')
            a = soporte.nueva_ejecucion(repo, datos, repo / 'ejecuciones')
            b = soporte.nueva_ejecucion(repo, datos, repo / 'ejecuciones')
            self.assertNotEqual(a, b)
            for destino in [datos, datos / 'hija', repo, repo / 'results/nueva']:
                with self.assertRaises(ValueError):
                    soporte.nueva_ejecucion(repo, datos, destino)
            alias = repo / 'alias'
            alias.symlink_to(datos, target_is_directory=True)
            with self.assertRaises(ValueError):
                soporte.nueva_ejecucion(repo, datos, alias / 'hija')
            self.assertEqual((datos / 'original').read_text(), 'sin cambios')

    def test_fuentes_sin_fallback_y_alias_editoriales_correctos(self):
        with tempfile.TemporaryDirectory() as tmp:
            salida, datos = Path(tmp) / 'ejecucion', Path(tmp) / 'datos'
            cat = soporte.catalogo_tablas(REPO, salida, datos)
            self.assertEqual(len(cat), 45)
            self.assertEqual(cat['S13']['estado'], 'documental')
            self.assertEqual(cat['S3']['estado'], soporte.ESTADOS[1])
            self.assertTrue(str(cat['S2E']['fuentes'][0]).endswith('tablaS2F_pool_calibracion.csv'))
            self.assertTrue(str(cat['S2F']['fuentes'][0]).endswith('tablaS2G_comprobacion_separada.csv'))
            for numero, entrada in cat.items():
                if entrada['estado'] == 'recalculado':
                    self.assertTrue(all(p.is_relative_to(salida) for p in entrada['fuentes']), numero)
            # Aunque existe la referencia distribuida, una ejecución incompleta falla.
            with self.assertRaises(FileNotFoundError):
                soporte.mostrar_tabla(REPO, salida, datos, cat, '2')

    def test_rezago_no_observable_no_se_presenta_como_cero(self):
        with tempfile.TemporaryDirectory() as tmp:
            salida = Path(tmp)
            tablas = salida / 'c50/c50/tablas'
            tablas.mkdir(parents=True)
            pd.DataFrame({'anio_ocur':[2022, 2023, 2024], '0':[80, 90, 100],
                          '1':[15, 10, 0], '2':[5, 0, 0]}).to_csv(tablas / 'tablaS_rezago_registro.csv', index=False)
            t = soporte.tabla_especial('S18', REPO, salida, Path(tmp) / 'datos')
            self.assertEqual(t['≥ 2 años (%)'].tolist(), [5.0, 'no observable', 'no observable'])

    def test_series_por_edad_conservadas_en_tendencia(self):
        with tempfile.TemporaryDirectory() as tmp:
            salida = Path(tmp)
            c = salida / 'c50/c50'
            (c / 'tablas').mkdir(parents=True)
            (c / 'reproduccion/pk/exploracion_previa').mkdir(parents=True)
            def fila(analisis, serie, valor):
                return dict(analisis=analisis, serie=serie, parametro='pendiente_pre',
                            estimacion_pct=valor, ic95_inf=valor-.1, ic95_sup=valor+.1)
            pd.DataFrame([fila('principal (control)', 'Nacional', 1),
                          fila('edad, modelo conjunto DK', '20-39', 2)]).to_csv(c / 'tablas/tablaS_tendencia_ampliaciones.csv', index=False)
            pd.DataFrame([fila('principal', 'Nacional', 1),
                          fila('edad_exploratorio', 'asmr_20-39', 3)]).to_csv(c / 'reproduccion/pk/exploracion_previa/estimaciones_ITS.csv', index=False)
            t = soporte.tabla_especial('S19', REPO, salida, Path(tmp) / 'datos')
            self.assertEqual(len(t), 3)
            self.assertIn('asmr_20-39', t['Serie'].tolist())
            self.assertEqual(t[t['Serie'] == 'asmr_20-39']['Pendiente previa, %/año'].iloc[0], '3.00 (2.90 a 3.10)')

    def test_nombre_entorno_no_exime_version_de_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / 'python'
            exe.touch()
            info = {'python':[3, 12, 3], 'paquetes':{
                'numpy':'2.3.5', 'pandas':'2.2.3', 'scipy':'1.18.1', 'statsmodels':'0.15.0',
                'matplotlib':'3.11.2', 'nbformat':'5.11.1', 'nbclient':'0.11.0',
                'ipykernel':'7.4.0', 'pypdfium2':'5.14.0'}}
            with patch.object(soporte.subprocess, 'run') as run:
                run.return_value.stdout = json.dumps(info)
                with self.assertRaisesRegex(RuntimeError, r'benchmark: Python 3.11 requerido'):
                    soporte.comprobar_entornos(REPO, exe, exe)

    def test_comparar_artistas_con_fechas_sin_perder_cambios(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        a, ax = plt.subplots()
        b, bx = plt.subplots()
        fechas = pd.date_range('2020-01-01', periods=3, freq='MS')
        ax.plot(fechas, [1, 2, 3])
        linea, = bx.plot(fechas, [1, 2, 3])
        self.assertEqual(soporte.artistas_numericos(a), soporte.artistas_numericos(b))
        linea.set_ydata([1, 2, 4])
        self.assertNotEqual(soporte.artistas_numericos(a), soporte.artistas_numericos(b))
        plt.close(a)
        plt.close(b)

    def test_detalle_de_discrepancias_conserva_tolerancias_originales(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'ref.json', Path(tmp) / 'ejec.json'
            a.write_text(json.dumps({'score_max_abs': 1e-10, 'resultado': 1000.0}))
            b.write_text(json.dumps({'score_max_abs': 2e-10, 'resultado': 1000.0 + 1e-9}))
            diferencias = soporte.detallar_diferencias(a, b, 'prueba.json')
            self.assertEqual([r['campo'] for r in diferencias], ['/score_max_abs'])
            self.assertEqual(diferencias[0]['referencia'], 1e-10)
            self.assertEqual(diferencias[0]['ejecucion'], 2e-10)


if __name__ == '__main__':
    unittest.main()
