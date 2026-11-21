import json
import os
import re
import urllib.request
from urllib.parse import quote
from net import opener

HF_BASE = "https://huggingface.co"
TIMEOUT = 15

# Model olmayan (veya tek parçalı) üst düzey yollar
RESERVED_PATHS = {
    "datasets", "spaces", "docs", "blog", "papers", "models", "collections", "settings",
    "organizations", "tasks", "learn", "posts", "join", "login", "pricing", "chat",
    "api", "new", "welcome", "enterprise",
}
# Kullanıcı/organizasyon profilinin alt sayfaları (huggingface.co/<sahip>/models gibi)
PROFILE_SUBPATHS = {"models", "datasets", "spaces", "collections", "papers"}

def _headers():
    headers = {"User-Agent": "Repo-Agent"}
    # Token çağrı anında okunur: load_dotenv'i çağıran modül import sonrası çalıştırır
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def parse_hf_model_id(text):
    """Link veya 'org/model' girdisinden 'org/model' döndürür; model linki değilse None."""
    if not text:
        return None
    text = text.strip()
    has_host = re.match(r'^(?:https?://)?(?:www\.)?huggingface\.co/', text, re.IGNORECASE)
    if has_host:
        text = text[has_host.end():]
    elif re.match(r'^[a-z][a-z0-9+.-]*://', text, re.IGNORECASE):
        return None  # başka bir siteye ait link
    text = re.split(r'[?#]', text)[0]
    parts = [p for p in text.split("/") if p]
    if len(parts) < 2:
        return None
    if parts[0].lower() in RESERVED_PATHS or parts[1].lower() in PROFILE_SUBPATHS:
        return None
    if not has_host and "." in parts[0]:
        return None  # 'github.com/a/b' gibi şemasız başka site yolları
    if not all(re.fullmatch(r'[\w.\-]+', p) and p.strip('.') for p in parts[:2]):
        return None  # geçersiz karakter ya da yalnız noktalardan oluşan parça ('..')
    return f"{parts[0]}/{parts[1]}"

def strip_frontmatter(text):
    """Model kartının başındaki '---' ile çevrili YAML frontmatter'ı ayıklar."""
    text = text.lstrip(chr(0xFEFF))  # baştaki BOM
    match = re.match(r'^---[ \t]*\r?\n(?:.*?\r?\n)?---[ \t]*(?:\r?\n|$)', text, re.DOTALL)
    if match:
        return text[match.end():].lstrip("\r\n")
    return text

def _build_description(data):
    parts = []
    if data.get("pipeline_tag"):
        parts.append(f"Görev: {data['pipeline_tag']}")
    if data.get("library_name"):
        parts.append(f"Kütüphane: {data['library_name']}")
    tags = [t for t in (data.get("tags") or []) if isinstance(t, str)][:5]
    if tags:
        parts.append(f"Etiketler: {', '.join(tags)}")
    if not parts:
        return "Açıklama bulunmuyor."
    # Serbest metinli alanlar (library_name, tags): LLM bağlamını şişirmesin diye kısa tutulur
    return ("Hugging Face modeli. " + "; ".join(parts) + ".")[:300]

def fetch_hf_model(model_id, with_readme=True):
    """(name, url, desc, readme, likes, downloads, pipeline_tag) döner; başarısızlıkta (None, None, None, None, 0, 0, None)."""
    api_url = f"{HF_BASE}/api/models/{quote(model_id, safe='/')}"
    try:
        req = urllib.request.Request(api_url, headers=_headers())
        with opener.open(req, timeout=TIMEOUT) as response:
            data = json.loads(response.read().decode('utf-8'))
        full_id = data["id"]
        desc = _build_description(data)
        likes = data.get("likes") or 0
        downloads = data.get("downloads") or 0
        pipeline_tag = data.get("pipeline_tag")
        # Kategori olarak Markdown başlıklarında gösterilir: beklenmedik karakter varsa kullanılmaz
        if not (isinstance(pipeline_tag, str) and re.fullmatch(r"[\w .\-+#/&]{1,40}", pipeline_tag)):
            pipeline_tag = None
        name = "hf:" + full_id
        url = f"{HF_BASE}/{full_id}"
        if not with_readme:
            return name, url, desc, None, likes, downloads, pipeline_tag
        try:
            readme_url = f"{HF_BASE}/{quote(full_id, safe='/')}/raw/main/README.md"
            req_readme = urllib.request.Request(readme_url, headers=_headers())
            with opener.open(req_readme, timeout=TIMEOUT) as rm:
                readme = strip_frontmatter(rm.read().decode('utf-8'))
        except Exception:
            readme = desc
        return name, url, desc, readme, likes, downloads, pipeline_tag
    except Exception:
        return None, None, None, None, 0, 0, None
