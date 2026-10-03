"""Pruebas ligeras: no usan datos del estudio salvo los hashes de results/reference, no descargan nada,
no requieren GPU ni credenciales. Los datos que se generan aquí son SINTÉTICOS (prefijo SINTETICO_) y no
tienen relación con las cifras del artículo.
Ejecutar: python -m unittest discover -s tests -v
"""
import csv, gzip, json, os, re, subprocess, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GMM = [sys.executable, str(REPO / 'scripts' / 'gmm.py')]
sys.path.insert(0, str(REPO / 'scripts'))


def correr(*args):
    env = os.environ.copy(); env.pop('PYTHONSAFEPATH', None)
    return subprocess.run(GMM + list(args), capture_output=True, text=True, env=env)


class TestCLI(unittest.TestCase):
    def test_ayuda(self):
        for sub in ['', 'verificar', 'verificar-archivo', 'benchmark', 'c50', 'figuras', 'comparar', 'entrenamiento']:
            r = correr(*( [sub, '--help'] if sub else ['--help'] ))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('usage', r.stdout.lower())

    def test_salida_no_puede_estar_en_referencia(self):
        with tempfile.TemporaryDirectory() as d:
            r = correr('benchmark', '--data-dir', d, '--output-dir', str(REPO / 'results/reference/x'))
            self.assertEqual(r.returncode, 2)

    def test_datos_ausentes_error_informativo(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
            r = correr('c50', '--data-dir', d, '--output-dir', str(Path(o) / 'ejec'))
            self.assertEqual(r.returncode, 2)
            self.assertIn('falta', r.stderr)

    def test_salida_no_vacia_rechazada(self):
        import gmm
        with tempfile.TemporaryDirectory() as o:
            (Path(o) / 'SINTETICO_archivo.txt').write_text('x')
            with self.assertRaises(gmm.ErrorUsuario):
                gmm.preparar_salida(Path(o))

    def test_referencia_integra(self):
        r = correr('verificar')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('PASS', r.stdout)


class TestConfiguracion(unittest.TestCase):
    def test_pasos_existen(self):
        c = json.loads((REPO / 'configs/pasos.json').read_text(encoding='utf-8'))
        for p in c['benchmark']['pasos']:
            self.assertTrue((REPO / 'src/benchmark' / p['script'].replace('codigo/', '', 1)).is_file(), p['script'])
        for s in c['c50']['pasos'] + [c['c50']['paso_figuras']['script']]:
            self.assertTrue((REPO / 'src/c50' / f'{s}.py').is_file(), s)

    def test_figuras_mapeadas(self):
        f = json.loads((REPO / 'configs/figuras.json').read_text(encoding='utf-8'))['figuras']
        self.assertEqual(sorted(f), sorted([f'Figura_{i}' for i in range(1, 7)] + [f'Figura_S{i}' for i in range(1, 7)]))
        for n, i in f.items():
            self.assertTrue((REPO / i['generador'].split(' ')[0]).is_file(), n)
            self.assertRegex(i['sha256_pdf_v13'], r'^[0-9a-f]{64}$')

    def test_datos_esperados(self):
        d = json.loads((REPO / 'configs/datos_esperados.json').read_text(encoding='utf-8'))['archivos']
        self.assertEqual(sum(1 for k in d if k.endswith('.npz')), 12)
        for k, v in d.items():
            self.assertRegex(v['sha256'], r'^[0-9a-f]{64}$', k)


class TestComparador(unittest.TestCase):
    def test_estados(self):
        from comparar_resultados import comparar, resumen
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            A, B = Path(a), Path(b)
            (A / 'SINTETICO_igual.csv').write_text('x,y\n1,2.5\n'); (B / 'SINTETICO_igual.csv').write_text('x,y\n1,2.5\n')
            (A / 'SINTETICO_tol.json').write_text(json.dumps({'v': 0.1 + 0.2})); (B / 'SINTETICO_tol.json').write_text(json.dumps({'v': 0.3}))
            (A / 'SINTETICO_dif.csv').write_text('x\n1.0\n'); (B / 'SINTETICO_dif.csv').write_text('x\n1.1\n')
            (A / 'SINTETICO.csv.gz').write_bytes(gzip.compress(b'x\n1\n', mtime=1)); (B / 'SINTETICO.csv.gz').write_bytes(gzip.compress(b'x\n1\n', mtime=2))
            filas = comparar(A, B, ['SINTETICO_igual.csv', 'SINTETICO_tol.json', 'SINTETICO_dif.csv', 'SINTETICO.csv.gz', 'SINTETICO_falta.csv'])
            est = {f['archivo']: f['estado'] for f in filas}
            self.assertEqual(est['SINTETICO_igual.csv'], 'identico_bytes')
            self.assertEqual(est['SINTETICO_tol.json'], 'coincide_tolerancia')
            self.assertEqual(est['SINTETICO_dif.csv'], 'difiere')
            self.assertEqual(est['SINTETICO.csv.gz'], 'identico_contenido')
            self.assertEqual(est['SINTETICO_falta.csv'], 'sin_referencia')


@unittest.skipUnless(__import__('importlib').util.find_spec('numpy'), 'requiere numpy')
class TestFormatoNPZ(unittest.TestCase):
    def test_detecta_forma_y_no_finitos(self):
        import numpy as np, gmm
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'benchmark/pronosticos/publicados'; p.mkdir(parents=True)
            x = np.zeros((3, 36, 3), dtype='float32'); x[0, 0, 0] = np.nan  # SINTÉTICO
            np.savez(p / 'fc_2014-12.npz', **{'Log-lineal': x})
            fallas = gmm._verificar_formatos(Path(d))
            self.assertTrue(any('forma' in f for f in fallas))
            self.assertTrue(any('no finitos' in f for f in fallas))
            self.assertTrue(any('modelos' in f for f in fallas))


class TestPublicable(unittest.TestCase):
    PATRON = re.compile(r'/Users/|/home/[a-z]|C:\\\\Users|\.claude-science')

    def test_sin_rutas_personales(self):
        malos = []
        for p in REPO.rglob('*'):
            if p.is_file() and p.suffix in {'.py', '.md', '.json', '.txt', '.cff', '.yml', '.csv', '.toml'} and 'tests' not in p.parts:
                if self.PATRON.search(p.read_text(encoding='utf-8', errors='ignore')):
                    malos.append(str(p.relative_to(REPO)))
        self.assertEqual(malos, [])


if __name__ == '__main__':
    unittest.main()
