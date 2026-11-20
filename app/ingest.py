import json
import urllib.request
from html.parser import HTMLParser
import re
from llama_cpp import Llama
from database import init_db, insert_repo, get_db
import sqlite3

import os
from dotenv import load_dotenv
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_PATH")

# Model yükleme
print("Nomic modeli yükleniyor...")
embedder = Llama(
    model_path=EMBEDDING_MODEL, 
    embedding=True, 
    n_gpu_layers=-1, 
    verbose=False
)

class BookmarkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.repos = set()
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href' and 'github.com' in value:
                    match = re.search(r'github\.com/([^/]+)/([^/#?]+)', value)
                    if match:
                        self.repos.add(f"{match.group(1)}/{match.group(2)}")

def fetch_github_info(repo_path):
    api_url = f"https://api.github.com/repos/{repo_path}"
    # Varsayılan daldaki README'yi (dal ve dosya adı fark etmeksizin) ham metin olarak döner
    readme_url = f"https://api.github.com/repos/{repo_path}/readme"
    
    headers = {"User-Agent": "Repo-Agent"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            # Bazen description null gelebilir, str() ile güvenceye alıyoruz
            desc = data.get("description")
            if desc is None:
                desc = "Açıklama bulunmuyor."
                
            stars = data.get("stargazers_count", 0)
            forks = data.get("forks_count", 0)
            
        try:
            req_readme = urllib.request.Request(readme_url, headers={**headers, "Accept": "application/vnd.github.raw"})
            with urllib.request.urlopen(req_readme) as rm:
                readme = rm.read().decode('utf-8')
        except:
            readme = desc 
            
        return data.get("full_name"), data.get("html_url"), desc, readme, stars, forks
    except Exception as e:
        # Hata durumunu konsolda görebilmek için yazdırıyoruz
        # print(f"API Hatası ({repo_path}): {e}") 
        return None, None, None, None, 0, 0

def repo_exists_in_db(repo_path):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM repos WHERE url LIKE ?", (f"%{repo_path}%",))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except:
        return False

if __name__ == "__main__":
    init_db()
    parser = BookmarkParser()
    with open("data/bookmarks.html", "r", encoding="utf-8", errors="ignore") as f:
        parser.feed(f.read())
        
    repos = list(parser.repos)
    print(f"{len(repos)} adet eşsiz repo bulundu. İşlem başlıyor...\n")
    
    for i, repo_path in enumerate(repos):
        if repo_exists_in_db(repo_path):
            print(f"[{i+1}/{len(repos)}] ⏭️ Zaten var, atlandı: {repo_path}")
            continue
            
        name, url, desc, readme, stars, forks = fetch_github_info(repo_path)
        if name:
            # Vektörleme için RAM şişmesin diye readme'nin sadece 1000 karakterini kullanıyoruz
            text_to_embed = f"Name: {name}. Description: {desc}. Context: {readme[:1000]}"
            vector = embedder.create_embedding(text_to_embed)["data"][0]["embedding"]
            
            # Veritabanına ise README'nin TAMAMINI ve istatistikleri kaydediyoruz
            insert_repo(name, url, desc, readme, stars, forks, vector)
            print(f"[{i+1}/{len(repos)}] ✅ Eklendi (⭐ {stars}): {name}")
        else:
            print(f"[{i+1}/{len(repos)}] ❌ Ulaşılamadı: {repo_path}")
            
    print("\nVeritabanı güncel! Artık arayüzü başlatabilirsiniz.")