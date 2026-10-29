import http.server, socketserver, os, functools
DIR = os.path.abspath(os.environ["IMPORT_DIR"])
class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
socketserver.TCPServer.allow_reuse_address = True
Handler = functools.partial(CORSRequestHandler, directory=DIR)
try:
    httpd = socketserver.TCPServer(("", 8081), Handler)
    httpd.serve_forever()
except Exception:
    pass
