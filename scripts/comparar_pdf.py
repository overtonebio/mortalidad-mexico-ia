"""Compara dos PDF (requiere pypdfium2 y numpy; environments/comparacion-figuras.txt) de figura: SHA-256, número de páginas, tamaño de página y diferencia de píxeles
al rasterizar a 150 ppp (pypdfium2). También compara el texto extraído. Uso: comparar_pdf.py A.pdf B.pdf"""
import hashlib, sys, json
from pathlib import Path
def comparar_pdf(a, b, dpi=150):
    import pypdfium2 as pdfium, numpy as np
    r = {'sha_igual': hashlib.sha256(Path(a).read_bytes()).hexdigest() == hashlib.sha256(Path(b).read_bytes()).hexdigest()}
    A, B = pdfium.PdfDocument(str(a)), pdfium.PdfDocument(str(b))
    r['paginas'] = (len(A), len(B))
    if len(A) != 1 or len(B) != 1:
        r['estado'] = 'figura_no_es_de_una_pagina'
        A.close(); B.close()
        return r
    pa, pb = A[0], B[0]
    r['tam_pt'] = ([round(v, 2) for v in pa.get_size()], [round(v, 2) for v in pb.get_size()])
    ta = pa.get_textpage().get_text_range(); tb = pb.get_textpage().get_text_range()
    r['texto_igual'] = ta == tb
    ia = np.asarray(pa.render(scale=dpi / 72).to_pil().convert('RGB'), dtype=np.int16)
    ib = np.asarray(pb.render(scale=dpi / 72).to_pil().convert('RGB'), dtype=np.int16)
    if ia.shape != ib.shape:
        r['raster'] = f'dimensiones distintas {ia.shape} vs {ib.shape}'
        A.close(); B.close()
        return r
    d = np.abs(ia - ib); r['pix_dif_frac'] = float((d > 0).mean()); r['pix_dif_max'] = int(d.max())
    A.close(); B.close()
    return r
if __name__ == '__main__': print(json.dumps(comparar_pdf(sys.argv[1], sys.argv[2])))
