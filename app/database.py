import sqlite3
import sqlite_vec
import struct
import math

DB_PATH = "data/database/repos.db" 

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS repos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            url TEXT,
            description TEXT,
            readme TEXT,
            stars INTEGER,
            forks INTEGER
        )
    ''')
    # Veritabanını silmeden yeni özellikleri ekleyen zeki kalkan
    try:
        conn.execute("ALTER TABLE repos ADD COLUMN is_pinned INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass 
        
    try:
        conn.execute("ALTER TABLE repos ADD COLUMN category TEXT DEFAULT 'Genel'")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE repos ADD COLUMN source TEXT DEFAULT 'github'")
    except sqlite3.OperationalError:
        pass

    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_repos USING vec0(
            embedding float[768]
        )
    ''')
    conn.commit()
    conn.close()

def insert_repo(name, url, description, readme, stars, forks, embedding_vector, category="Genel", source="github"):
    conn = get_db()
    try:
        cursor = conn.cursor()
        # Kategori sütununu da ekleyerek kaydediyoruz
        cursor.execute(
            "INSERT INTO repos (name, url, description, readme, stars, forks, is_pinned, category, source) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (name, url, description, readme, stars, forks, category, source)
        )
        row_id = cursor.lastrowid
        
        while isinstance(embedding_vector, list) and len(embedding_vector) > 0 and isinstance(embedding_vector[0], list):
            embedding_vector = embedding_vector[0]
            
        clean_vector = [float(x) for x in embedding_vector]
        vector_bytes = struct.pack(f"{len(clean_vector)}f", *clean_vector)
        
        cursor.execute(
            "INSERT INTO vec_repos (rowid, embedding) VALUES (?, ?)",
            (row_id, vector_bytes)
        )
        conn.commit()
        return True 
    except sqlite3.IntegrityError:
        return False 
    finally:
        conn.close()

def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

def search_repos(query_text, query_vector, limit=3, source=None):
    conn = get_db()
    vector_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

    if source:
        # Kaynak süzgeci KNN'den sonra uygulanır; adaylar elenmesin diye tüm vektörler taranır
        k = min(conn.execute("SELECT COUNT(*) FROM vec_repos").fetchone()[0], 4096) or 1
        cursor = conn.execute('''
            SELECT r.name, r.url, r.description, r.readme, r.stars, v.embedding
            FROM vec_repos v
            JOIN repos r ON r.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ? AND r.source = ?
        ''', (vector_bytes, k, source))
    else:
        cursor = conn.execute('''
            SELECT r.name, r.url, r.description, r.readme, r.stars, v.embedding
            FROM vec_repos v
            JOIN repos r ON r.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
        ''', (vector_bytes, 10))
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return []
        
    query_words = [w.lower() for w in query_text.split() if len(w) > 2]
    scored_results = []
    
    for r in results:
        name, url, desc, readme, stars, db_vector_bytes = r
        db_vector = struct.unpack(f"{len(query_vector)}f", db_vector_bytes)
        
        cos_sim = cosine_similarity(query_vector, db_vector)
        if cos_sim < 0: cos_sim = 0.0
        
        full_text = f"{name} {desc} {readme}".lower()
        kw_count = sum(full_text.count(w) for w in query_words)
        kw_boost = kw_count * 0.015 
        
        star_boost = math.log10(stars + 1) * 0.02
        final_score = cos_sim + kw_boost + star_boost
        
        uyum_yuzdesi = min(100, int(final_score * 100))
        
        scored_results.append({
            "repo_data": (name, url, desc, readme, uyum_yuzdesi),
            "uyum_yuzdesi": uyum_yuzdesi
        })
        
    scored_results.sort(key=lambda x: x["uyum_yuzdesi"], reverse=True)
    return [item["repo_data"] for item in scored_results[:limit]]

def get_all_repos():
    conn = get_db()
    # Çekerken 9. parametre category, 10. parametre source (github/huggingface)
    cursor = conn.execute("SELECT id, name, url, description, readme, stars, forks, is_pinned, category, source FROM repos ORDER BY is_pinned DESC, stars DESC")
    results = cursor.fetchall()
    conn.close()
    return results

def toggle_pin(repo_id, current_status):
    conn = get_db()
    new_status = 0 if current_status == 1 else 1
    conn.execute("UPDATE repos SET is_pinned = ? WHERE id = ?", (new_status, repo_id))
    conn.commit()
    conn.close()

def delete_repo(repo_id):
    """Hem veritabanından hem de vektör uzayından siler."""
    conn = get_db()
    conn.execute("DELETE FROM repos WHERE id = ?", (repo_id,))
    conn.execute("DELETE FROM vec_repos WHERE rowid = ?", (repo_id,))
    conn.commit()
    conn.close()

def update_repo_stats(repo_id, stars, forks):
    """Repo yıldız ve fork'larını günceller."""
    conn = get_db()
    conn.execute("UPDATE repos SET stars = ?, forks = ? WHERE id = ?", (stars, forks, repo_id))
    conn.commit()
    conn.close()

def update_repo_category(repo_id, new_category):
    """Reponun kategorisini günceller (Mevcut olanı seçer veya yeni oluşturur)."""
    conn = get_db()
    conn.execute("UPDATE repos SET category = ? WHERE id = ?", (new_category, repo_id))
    conn.commit()
    conn.close()

def get_all_categories(source=None):
    """Açılır liste (dropdown) için benzersiz kategorileri getirir; source verilirse yalnız o kaynağınkileri."""
    conn = get_db()
    if source:
        cursor = conn.execute("SELECT DISTINCT category FROM repos WHERE category IS NOT NULL AND source = ? ORDER BY category", (source,))
    else:
        cursor = conn.execute("SELECT DISTINCT category FROM repos WHERE category IS NOT NULL ORDER BY category")
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results

def normalize_category(kategori):
    """Kategoriyi mevcut olanlarla harf büyüklüğüne duyarsız eşleştirir; yoksa yalnız ilk harfi büyütür."""
    # Aynı kategori farklı harf büyüklüğüyle zaten varsa onu kullan (Örn: 'Ai' -> 'AI')
    for mevcut in get_all_categories():
        if mevcut.lower() == kategori.lower():
            return mevcut
    # .capitalize() 'AI' -> 'Ai' yaptığı için sadece ilk harf büyütülür
    return kategori[0].upper() + kategori[1:]

def rename_category_globally(old_category, new_category, source=None):
    """Belirli bir kategorinin adını topluca değiştirir (Örn: 'Ai' -> 'AI'); source verilirse yalnız o kaynakta."""
    conn = get_db()
    if source:
        conn.execute("UPDATE repos SET category = ? WHERE category = ? AND source = ?", (new_category, old_category, source))
    else:
        conn.execute("UPDATE repos SET category = ? WHERE category = ?", (new_category, old_category))
    conn.commit()
    conn.close()