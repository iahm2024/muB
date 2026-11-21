import urllib.request
from urllib.parse import urlparse

def _origin(url):
    parsed = urlparse(url)
    return parsed.scheme, parsed.hostname, parsed.port or {"http": 80, "https": 443}.get(parsed.scheme)

class _SameHostAuthRedirect(urllib.request.HTTPRedirectHandler):
    """urllib yönlendirmede Authorization başlığını taşır; şema, host veya port değişirse token'ı düşürür."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None and _origin(newurl) != _origin(req.full_url):
            new_req.remove_header("Authorization")
        return new_req

# Token taşıyan tüm GitHub ve Hugging Face istekleri bu opener ile yapılır
opener = urllib.request.build_opener(_SameHostAuthRedirect)
