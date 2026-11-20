import streamlit as st
import urllib.request
import json
import re
from collections import defaultdict
from llama_cpp import Llama
from database import search_repos, get_all_repos, insert_repo, toggle_pin, init_db, update_repo_stats, delete_repo, update_repo_category, get_all_categories, rename_category_globally

import os
from dotenv import load_dotenv
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_PATH")
LLM_MODEL = os.getenv("LLM_MODEL_PATH")

# Uygulama açılır açılmaz veritabanını kontrol et
init_db()

st.set_page_config(page_title="muB (Managable Useful Bookmarks)", page_icon="🧠", layout="wide")

# Modelleri önbelleğe alan fonksiyon.
# @st.cache_resource sayesinde sadece İLK çağrıldığında yüklenir, sonraki aramalarda hafızadan okunur.
@st.cache_resource
def load_models():
    embed_model = Llama(model_path=EMBEDDING_MODEL, embedding=True, verbose=False)
    llm = Llama(
        model_path=LLM_MODEL,
        n_ctx=4096, 
        n_gpu_layers=-1, 
        verbose=False
    )
    return embed_model, llm

@st.dialog("📄 README Dosyası", width="large")
def show_readme_dialog(repo_name, readme_content):
    st.markdown(f"### {repo_name}")
    st.divider()
    with st.container(height=600):
        st.markdown(readme_content, unsafe_allow_html=True)

def fetch_single_repo(repo_path, with_readme=True):
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
            desc = data.get("description") or "Açıklama bulunmuyor."
            stars = data.get("stargazers_count", 0)
            forks = data.get("forks_count", 0)
        if not with_readme:
            return data.get("full_name"), data.get("html_url"), desc, None, stars, forks
        try:
            req_readme = urllib.request.Request(readme_url, headers={**headers, "Accept": "application/vnd.github.raw"})
            with urllib.request.urlopen(req_readme) as rm:
                readme = rm.read().decode('utf-8')
        except:
            readme = desc
        return data.get("full_name"), data.get("html_url"), desc, readme, stars, forks
    except Exception as e:
        return None, None, None, None, 0, 0

def fetch_user_public_repos(username):
    api_url = f"https://api.github.com/search/repositories?q=user:{username}&sort=stars&order=desc&per_page=100"
    headers = {"User-Agent": "Repo-Agent"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            repos = [{"full_name": r["full_name"], "stars": r["stargazers_count"]} for r in data.get("items", [])]
            return repos
    except Exception as e:
        return None

def auto_categorize_repo(llm, name, desc):
    """Qwen LLM kullanarak repoya tek kelimelik kategori atar."""
    prompt = f"""<|im_start|>system
Sen bir kategorizasyon asistanısın. Görevin verilen GitHub reposunu (Örn: Frontend, Backend, AI, Veritabanı, DevOps, Framework, Tool, Eğitim vb.) gibi TEK BİR KELİME ile kategorize etmektir.
SADECE kategori adını yaz, nokta bile koyma.
<|im_end|>
<|im_start|>user
Repo Adı: {name}
Açıklama: {desc}
<|im_end|>
<|im_start|>assistant
"""
    response = llm(prompt, max_tokens=10, temperature=0.1, stop=["<|im_end|>"])
    kategori = response["choices"][0]["text"].strip()

    # Model halüsinasyon görüp uzun cümle kurarsa varsayılan'a dön
    if len(kategori) > 15 or len(kategori) < 2:
        return "Genel"

    # Aynı kategori farklı harf büyüklüğüyle zaten varsa onu kullan (Örn: 'Ai' -> 'AI')
    for mevcut in get_all_categories():
        if mevcut.lower() == kategori.lower():
            return mevcut
    # .capitalize() 'AI' -> 'Ai' yaptığı için sadece ilk harf büyütülür
    return kategori[0].upper() + kategori[1:]

def generate_markdown_table(repos):
    md_content = "# 📚 Kaydedilen GitHub Repolarım\n\n"
    md_content += "| Repo Adı | 🏷️ Kategori | ⭐ Yıldız | 🔗 Link | 📝 Açıklama |\n"
    md_content += "|---|---|---|---|---|\n"
    for r in repos:
        r_id, name, url, desc, readme, stars, forks, is_pinned, category = r
        clean_desc = (desc or "Açıklama bulunmuyor.").replace('\n', ' ').replace('\r', '').replace('|', '-')
        md_content += f"| **{name}** | {category} | {stars} | [Git]({url}) | {clean_desc} |\n"
    return md_content

# --- OTURUM (SESSION) YÖNETİMİ ---
if "aktif_sayfa" not in st.session_state:
    st.session_state.aktif_sayfa = "Liste"
if "user_repos_cache" not in st.session_state:
    st.session_state.user_repos_cache = []

# --- YAN MENÜ ---
st.sidebar.title("🧭 Navigasyon")
if st.sidebar.button("📚 Tüm Repoları İncele", use_container_width=True):
    st.session_state.aktif_sayfa = "Liste"
if st.sidebar.button("🧠 Yapay Zeka ile Ara", use_container_width=True):
    st.session_state.aktif_sayfa = "YapayZeka"
st.sidebar.divider()
st.sidebar.markdown("muB - Yer imlerinizi akıllıca yönetin.")

# ==========================================
# SAYFA 1: TÜM REPOLAR
# ==========================================
if st.session_state.aktif_sayfa == "Liste":
    st.title("📚 Kaydedilen Tüm Repolar")
    
    # 🌟 YENİLİK: TEK TIKLA GÜNCELLEME BUTONU
    with st.expander("🔄 Tüm Listeyi GitHub'dan Güncelle", expanded=False):
        st.markdown("Veritabanınızdaki tüm repoların Yıldız ve Fork sayılarını GitHub'dan en güncel haliyle çeker.")
        if st.button("Tüm İstatistikleri Senkronize Et", use_container_width=True):
            tum_repolar = get_all_repos()
            if tum_repolar:
                basari_sayisi = 0
                with st.spinner("Tüm repolar güncelleniyor... (Bu işlem sayısına göre 1-2 dakika sürebilir)"):
                    for r in tum_repolar:
                        r_id, name, url, desc, readme, stars, forks, is_pinned, category = r
                        _, _, _, _, n_stars, n_forks = fetch_single_repo(name, with_readme=False)
                        if n_stars > 0 or n_forks > 0:
                            update_repo_stats(r_id, n_stars, n_forks)
                            basari_sayisi += 1
                st.success(f"✅ {basari_sayisi} reponun istatistikleri başarıyla güncellendi!")
                st.rerun()

    with st.expander("➕ Sisteme Tek Bir Repo Ekle", expanded=False):
        yeni_repo_url = st.text_input("GitHub Repo Linkini Yapıştırın:", placeholder="Örn: https://github.com/facebook/react")
        if st.button("Repoyu Ekle ve Vektörleştir"):
            if yeni_repo_url:
                match = re.search(r'(?:github\.com/)?([^/]+)/([^/#?]+)', yeni_repo_url)
                if match:
                    repo_path = f"{match.group(1)}/{match.group(2)}"
                    with st.spinner(f"'{repo_path}' analiz ediliyor ve AI ile kategorize ediliyor..."):
                        name, url, desc, readme, stars, forks = fetch_single_repo(repo_path)
                        if name:
                            embedder, llm = load_models()
                            
                            # AI ile Kategori Belirle
                            kategori = auto_categorize_repo(llm, name, desc)
                            
                            text_to_embed = f"Name: {name}. Description: {desc}. Context: {readme[:1000]}"
                            vector = embedder.create_embedding(text_to_embed)["data"][0]["embedding"]
                            
                            basari = insert_repo(name, url, desc, readme, stars, forks, vector, kategori)
                            if basari:
                                st.success(f"✅ Eklendi! (Kategori: {kategori})")
                                st.rerun() 
                            else:
                                st.warning("⚠️ Zaten kayıtlı.")
                        else:
                            st.error("❌ Repo bulunamadı.")
                else:
                    st.error("❌ Geçersiz link.")

    with st.expander("👤 Bir Kullanıcının Repolarını Toplu Seç", expanded=False):
        col_usr, col_btn = st.columns([3, 1])
        with col_usr:
            kullanici_adi = st.text_input("Kullanıcı/Organizasyon Adı:", placeholder="Örn: karpathy, facebook, microsoft...")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Repolarını Getir", use_container_width=True):
                if kullanici_adi:
                    with st.spinner("Kullanıcının public repoları listeleniyor..."):
                        fetched = fetch_user_public_repos(kullanici_adi)
                        if fetched is not None:
                            st.session_state.user_repos_cache = fetched
                        else:
                            st.error("❌ Kullanıcı bulunamadı veya API limiti aşıldı.")

        if st.session_state.user_repos_cache:
            options = [f"⭐ {r['stars']} | {r['full_name']}" for r in st.session_state.user_repos_cache]
            secilenler = st.multiselect(
                "Sisteme Eklemek İstediklerinizi İşaretleyin:",
                options=options,
                default=[]
            )
            if st.button("Seçilenleri Veritabanına Ekle"):
                if secilenler:
                    embedder, llm = load_models()
                    basarili_sayisi = 0
                    for secim in secilenler:
                        repo_path = secim.split(" | ")[1]
                        with st.spinner(f"'{repo_path}' indiriliyor, AI ile kategorize ediliyor..."):
                            name, url, desc, readme, stars, forks = fetch_single_repo(repo_path)
                            if name:
                                kategori = auto_categorize_repo(llm, name, desc)
                                text_to_embed = f"Name: {name}. Description: {desc}. Context: {readme[:1000]}"
                                vector = embedder.create_embedding(text_to_embed)["data"][0]["embedding"]
                                if insert_repo(name, url, desc, readme, stars, forks, vector, kategori):
                                    st.toast(f"✅ {name} ({kategori}) eklendi!")
                                    basarili_sayisi += 1
                                else:
                                    st.toast(f"⚠️ {name} zaten vardı.")
                    st.session_state.user_repos_cache = [] 
                    st.success(f"İşlem tamamlandı! {basarili_sayisi} yeni repo eklendi.")
                    st.rerun()

    tum_repolar = get_all_repos()
    
    if not tum_repolar:
        st.warning("Veritabanı boş.")
    else:
        st.divider()
        col_arama, col_grup = st.columns([2, 1])
        with col_arama:
            search_query = st.text_input("🔍 Listede İsim veya Açıklama Ara...", placeholder="Örn: react, database, ui...")
        with col_grup:
            st.markdown("<br>", unsafe_allow_html=True)
            gorunum = st.radio("Listeleme Görünümü:", ["Varsayılan", "🧑‍💻 Yazara Göre", "🏷️ Kategoriye Göre"], horizontal=True, label_visibility="collapsed")
        
        if search_query:
            tum_repolar = [r for r in tum_repolar if search_query.lower() in r[1].lower() or (r[3] and search_query.lower() in r[3].lower())]
            
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(f"*(Toplam **{len(tum_repolar)}** repo listeleniyor)*")
        with col_btn:
            md_table_string = generate_markdown_table(tum_repolar)
            st.download_button("📥 Listeyi İndir (.md)", data=md_table_string, file_name="favori_repolarim.md", mime="text/markdown", use_container_width=True)
        
        st.divider()

        mevcut_kategoriler = get_all_categories()

        def draw_repo(r):
            r_id, name, url, desc, readme, stars, forks, is_pinned, category = r
            pin_ikon = "📌 " if is_pinned else ""
            
            with st.expander(f"{pin_ikon}{name}  —  [{category}]"):
                c1, c2, c3 = st.columns([5, 1, 1])
                with c1:
                    st.markdown(f"**⭐ {stars}** &nbsp; | &nbsp; **🔀 {forks}** &nbsp; | &nbsp; 🔗 **[GitHub'da Aç ↗]({url})**")
                    st.markdown(f"*{desc}*")
                with c2:
                    if st.button("📍 Kaldır" if is_pinned else "📌 Sabitle", key=f"pin_{r_id}", use_container_width=True):
                        toggle_pin(r_id, is_pinned)
                        st.rerun() 
                with c3:
                    if st.button("🗑️ Sil", key=f"del_{r_id}", type="primary", use_container_width=True):
                        delete_repo(r_id)
                        st.rerun()

                # --- KATEGORİ SEÇME (DROPDOWN) VE YENİ EKLEME BÖLÜMÜ ---
                with st.container():
                    kat_c1, kat_c2, kat_c3 = st.columns([3, 2, 1])
                    with kat_c1:
                        # Eğer reponun kategorisi listede yoksa ekleyelim, ayrıca "Yeni Ekle..." seçeneği koyalım
                        options = list(set(mevcut_kategoriler + [category]))
                        options.sort() # Alfabetik dizersin
                        options.append("➕ Yeni Ekle...")
                        
                        secilen_kat = st.selectbox("🏷️ Kategori Seç:", options=options, index=options.index(category), key=f"kat_select_{r_id}")
                    
                    with kat_c2:
                        yeni_kategori = ""
                        # Kullanıcı listeden seçmek yerine "Yeni Ekle" derse metin kutusu açılır
                        if secilen_kat == "➕ Yeni Ekle...":
                            yeni_kategori = st.text_input("Yeni Kategori Adı:", key=f"kat_input_{r_id}").strip()
                    
                    with kat_c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("💾 Kaydet", key=f"kat_btn_{r_id}", use_container_width=True):
                            hedef_kategori = yeni_kategori if secilen_kat == "➕ Yeni Ekle..." else secilen_kat
                            if hedef_kategori and hedef_kategori != category:
                                update_repo_category(r_id, hedef_kategori)
                                st.rerun()

                st.divider()
                if st.button("📖 README Dosyasını Oku", key=f"read_btn_{r_id}", use_container_width=True):
                    show_readme_dialog(name, readme)

        if gorunum == "Varsayılan":
            for r in tum_repolar:
                draw_repo(r)
        elif gorunum == "🧑‍💻 Yazara Göre":
            grouped = defaultdict(list)
            for r in tum_repolar:
                match = re.search(r'github\.com/([^/]+)/', r[2])
                owner = match.group(1) if match else "Diğer"
                grouped[owner].append(r)
            for owner, repos in grouped.items():
                st.markdown(f"### 🧑‍💻 {owner} ({len(repos)} repo)")
                for r in repos:
                    draw_repo(r)
        elif gorunum == "🏷️ Kategoriye Göre":
            grouped = defaultdict(list)
            for r in tum_repolar:
                kategori = r[8] 
                grouped[kategori].append(r)
                
            for kat, repos in grouped.items():
                # 3 yerine sadece 2 kolon: Geniş bir başlık alanı ve zarif bir buton alanı
                kat_col1, kat_col2 = st.columns([8, 2])
                with kat_col1:
                    st.markdown(f"### 🏷️ {kat} ({len(repos)} repo)")
                with kat_col2:
                    st.markdown("<br>", unsafe_allow_html=True) # Başlıkla dikey hizalamak için
                    
                    # st.popover minik, şık bir buton oluşturur, tıklanınca içindekileri gösterir
                    with st.popover("✏️ Kategoriyi Düzenle", use_container_width=True):
                        st.markdown(f"**'{kat}' Kategorisini Değiştir**")
                        yeni_kat_adi = st.text_input("Yeni İsim:", value=kat, key=f"ren_in_{kat}", label_visibility="collapsed")
                        
                        if st.button("🔄 Tümünü Güncelle", key=f"ren_btn_{kat}", use_container_width=True):
                            if yeni_kat_adi.strip() and yeni_kat_adi.strip() != kat:
                                rename_category_globally(kat, yeni_kat_adi.strip())
                                st.rerun()
                
                # O kategoriye ait repoları çiz
                for r in repos:
                    draw_repo(r)

# SAYFA 2: YAPAY ZEKA ARAMASI
elif st.session_state.aktif_sayfa == "YapayZeka":
    st.title("🧠 Yapay Zeka ile Arama")
    st.markdown("Kaydettiğiniz repolar arasında anlamsal arama yapın.")

    query = st.text_input("Ne arıyorsunuz?", placeholder="Örn: Frontend arayüz tasarlamak için frameworkler...")

    if query:
        with st.spinner("Modeller uyanıyor..."):
            embedder, llm = load_models()
            
        with st.spinner("Hibrit arama (Vektör + Kelime + Yıldız) yapılıyor..."):
            query_vector = embedder.create_embedding(query)["data"][0]["embedding"]
            results = search_repos(query, query_vector, limit=3)
            
            if not results:
                st.warning("Veritabanında uygun repo bulunamadı.")
            else:
                context = ""
                for r in results:
                    name, url, desc, readme, uyum_yuzdesi = r
                    context += f"\nRepo Adı: {name}\nURL: {url}\nAçıklama: {desc}\nREADME Özeti: {readme[:1500]}\n"

                prompt = f"""<|im_start|>system
Sen kullanıcının yer imlerini analiz eden DİSİPLİNLİ ve KATI bir asistansın.
KESİN KURALLAR:
1. SADECE VE SADECE aşağıdaki 'Bulunan Repolar' bağlamında yer alan bilgileri kullanarak cevap ver.
2. Kendi genel bilgilerini ASLA KULLANMA. Örneğin sana bir kişi sorulursa, o kişinin biyografisini veya başarılarını anlatma. Sadece bağlamda o kişiye ait hangi repolar varsa, onların ne işe yaradığını açıkla.
3. Repolar soruyla tam uyuşmuyorsa bile "Tam eşleşme yok ancak elimdeki repolar şunlar:" diyerek SADECE BAĞLAMDAKİLERİ tanıt.

Bağlam:
{context}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
                response = llm(prompt, max_tokens=512, temperature=0.1, stop=["<|im_end|>"])
                answer = response["choices"][0]["text"]
                
                st.markdown("### 🤖 Qwen Yanıtı")
                st.success(answer.strip())
                
                st.divider()
                st.markdown("#### 📂 Referans Alınan Kaynaklar")
                for r in results:
                    name, url, desc, readme, uyum_yuzdesi = r
                    st.markdown(f"- **[{name}]({url})** *(% {uyum_yuzdesi} Uyum)*\n  > {desc}")