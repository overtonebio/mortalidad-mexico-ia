import io, zipfile, requests, sys
class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url; h = requests.head(url, allow_redirects=True, timeout=60); self.size = int(h.headers['Content-Length']); self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b''
        end = min(self.pos + n, self.size) - 1
        r = requests.get(self.url, headers={'Range': f'bytes={self.pos}-{end}'}, timeout=120); b = r.content; self.pos += len(b); return b
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
def abrir(url): return zipfile.ZipFile(io.BufferedReader(HTTPFile(url), buffer_size=1 << 16))
