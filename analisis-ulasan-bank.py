import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Set style seaborn
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 14})

# Fungsi untuk membaca dataset
def load_data(file_path):
    """
    Membaca dataset dari file CSV atau Excel
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel")
    
    print(f"Dataset berhasil dimuat dengan {df.shape[0]} baris dan {df.shape[1]} kolom")
    return df

# Fungsi untuk konversi format waktu
def convert_datetime_format(df, date_column):
    """
    Mengonversi kolom tanggal ke format DD/MM/YYYY
    """
    # Deteksi format tanggal asli dan konversi
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df[date_column + '_formatted'] = df[date_column].dt.strftime('%d/%m/%Y')
        print(f"Kolom {date_column} berhasil dikonversi ke format DD/MM/YYYY")
    except Exception as e:
        print(f"Error saat mengonversi tanggal: {str(e)}")
    
    return df

# 1. Analisis distribusi rating
def analyze_rating_distribution(df, rating_column):
    """
    Menganalisis distribusi rating (1-5 bintang)
    """
    plt.figure(figsize=(10, 6))
    rating_counts = df[rating_column].value_counts().sort_index()
    
    # Bar chart
    ax = sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
    
    # Tambahkan label di atas bar
    for i, count in enumerate(rating_counts.values):
        ax.text(i, count + 5, f"{count}", ha='center')
    
    plt.title('Distribusi Rating Ulasan Bank')
    plt.xlabel('Rating (Bintang)')
    plt.ylabel('Jumlah Ulasan')
    plt.tight_layout()
    plt.savefig('rating_distribution.png')
    plt.show()
    
    # Hitung persentase
    total = rating_counts.sum()
    percentages = (rating_counts / total * 100).round(2)
    
    print("Distribusi Rating:")
    for rating, count in rating_counts.items():
        print(f"Rating {rating} bintang: {count} ulasan ({percentages[rating]}%)")
    
    return rating_counts

# 2. Analisis data kosong
def analyze_missing_reviews(df, review_column):
    """
    Menganalisis berapa banyak ulasan yang kosong
    """
    missing_reviews = df[review_column].isna().sum()
    empty_reviews = df[df[review_column].str.strip() == ""].shape[0] if isinstance(df[review_column].iloc[0], str) else 0
    total_missing = missing_reviews + empty_reviews
    percentage = (total_missing / len(df) * 100).round(2)
    
    plt.figure(figsize=(8, 6))
    plt.pie([total_missing, len(df) - total_missing], 
            labels=['Kosong', 'Terisi'], 
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff'],
            startangle=90)
    plt.axis('equal')
    plt.title('Persentase Ulasan Kosong vs Terisi')
    plt.tight_layout()
    plt.savefig('missing_reviews.png')
    plt.show()
    
    print(f"\nAnalisis Ulasan Kosong:")
    print(f"- Ulasan NULL: {missing_reviews}")
    print(f"- Ulasan kosong (string kosong): {empty_reviews}")
    print(f"- Total ulasan kosong: {total_missing} dari {len(df)} ({percentage}%)")
    
    return total_missing

# 3. Analisis panjang ulasan
def analyze_review_length(df, review_column):
    """
    Menganalisis distribusi panjang ulasan (kata dan karakter)
    """
    # Filter ulasan yang tidak kosong
    df_valid = df.dropna(subset=[review_column])
    df_valid = df_valid[df_valid[review_column].str.strip() != ""]
    
    # Hitung panjang kata dan karakter
    df_valid['word_count'] = df_valid[review_column].apply(lambda x: len(str(x).split()))
    df_valid['char_count'] = df_valid[review_column].apply(lambda x: len(str(x)))
    
    # Statistik deskriptif
    word_stats = df_valid['word_count'].describe().round(2)
    char_stats = df_valid['char_count'].describe().round(2)
    
    # Plot histogram untuk jumlah kata
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    sns.histplot(df_valid['word_count'], kde=True, bins=30)
    plt.title('Distribusi Jumlah Kata dalam Ulasan')
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Frekuensi')
    plt.axvline(word_stats['mean'], color='r', linestyle='--', label=f"Rata-rata: {word_stats['mean']}")
    plt.axvline(word_stats['50%'], color='g', linestyle='--', label=f"Median: {word_stats['50%']}")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    sns.histplot(df_valid['char_count'], kde=True, bins=30)
    plt.title('Distribusi Jumlah Karakter dalam Ulasan')
    plt.xlabel('Jumlah Karakter')
    plt.ylabel('Frekuensi')
    plt.axvline(char_stats['mean'], color='r', linestyle='--', label=f"Rata-rata: {char_stats['mean']}")
    plt.axvline(char_stats['50%'], color='g', linestyle='--', label=f"Median: {char_stats['50%']}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('review_length_distribution.png')
    plt.show()
    
    print("\nStatistik Panjang Ulasan:")
    print("Jumlah Kata:")
    for stat, value in word_stats.items():
        print(f"- {stat}: {value}")
    
    print("\nJumlah Karakter:")
    for stat, value in char_stats.items():
        print(f"- {stat}: {value}")
    
    return df_valid[['word_count', 'char_count']]

# 4. Analisis tren waktu
def analyze_time_trends(df, date_column):
    """
    Menganalisis tren jumlah ulasan berdasarkan waktu
    """
    # Pastikan kolom tanggal dalam format datetime
    if date_column + '_formatted' in df.columns:
        date_col = date_column + '_formatted'
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
    else:
        date_col = date_column
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Buat kolom tahun dan bulan
    df['tahun'] = df[date_col].dt.year
    df['bulan'] = df[date_col].dt.month
    df['tahun_bulan'] = df[date_col].dt.strftime('%Y-%m')
    
    # Analisis bulanan
    monthly_counts = df.groupby('tahun_bulan').size().reset_index(name='jumlah')
    monthly_counts['tahun_bulan'] = pd.to_datetime(monthly_counts['tahun_bulan'])
    monthly_counts = monthly_counts.sort_values('tahun_bulan')
    
    # Analisis tahunan
    yearly_counts = df.groupby('tahun').size().reset_index(name='jumlah')
    yearly_counts = yearly_counts.sort_values('tahun')
    
    # Plot tren bulanan
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(monthly_counts['tahun_bulan'], monthly_counts['jumlah'], marker='o', linestyle='-')
    plt.title('Tren Jumlah Ulasan per Bulan')
    plt.xlabel('Bulan')
    plt.ylabel('Jumlah Ulasan')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.bar(yearly_counts['tahun'].astype(str), yearly_counts['jumlah'], color='skyblue')
    plt.title('Tren Jumlah Ulasan per Tahun')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Ulasan')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Tambahkan label jumlah di atas bar
    for i, count in enumerate(yearly_counts['jumlah']):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('time_trends.png')
    plt.show()
    
    print("\nTren Jumlah Ulasan:")
    print("Jumlah Ulasan per Tahun:")
    for _, row in yearly_counts.iterrows():
        print(f"- {row['tahun']}: {row['jumlah']} ulasan")
    
    return {'monthly': monthly_counts, 'yearly': yearly_counts}

# 5. Analisis frekuensi kata
def analyze_word_frequency(df, review_column, top_n=20):
    """
    Menganalisis frekuensi kata dalam ulasan
    """
    # Filter ulasan yang tidak kosong
    df_valid = df.dropna(subset=[review_column])
    df_valid = df_valid[df_valid[review_column].str.strip() != ""]
    
    # Gabungkan semua ulasan
    all_reviews = ' '.join(df_valid[review_column].astype(str))
    
    # Tokenisasi
    stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
    
    # Tambahkan kata-kata umum yang tidak bermakna
    additional_stopwords = ['bank', 'nya', 'yg', 'di', 'dan', 'dengan', 'untuk', 'ini', 'itu', 'saya', 'ke', 'dari']
    stop_words.update(additional_stopwords)
    
    # Bersihkan teks dan tokenisasi
    all_reviews = re.sub(r'[^\w\s]', ' ', all_reviews.lower())
    tokens = word_tokenize(all_reviews)
    
    # Filter stopwords dan kata pendek
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Hitung frekuensi
    word_freq = Counter(filtered_tokens)
    most_common = word_freq.most_common(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    words, counts = zip(*most_common)
    plt.barh(words, counts, color='skyblue')
    plt.gca().invert_yaxis()  # Membalik urutan untuk kata teratas di atas
    plt.title(f'Top {top_n} Kata yang Sering Muncul dalam Ulasan')
    plt.xlabel('Frekuensi')
    plt.ylabel('Kata')
    plt.tight_layout()
    plt.savefig('word_frequency.png')
    plt.show()
    
    print(f"\nTop {top_n} Kata yang Sering Muncul:")
    for word, count in most_common:
        print(f"- {word}: {count} kali")
    
    return word_freq

# 6. Deteksi duplikasi ulasan
def detect_duplicate_reviews(df, review_column, threshold=0.8):
    """
    Mendeteksi ulasan yang sama atau sangat mirip menggunakan cosine similarity
    """
    # Filter ulasan yang tidak kosong
    df_valid = df.dropna(subset=[review_column])
    df_valid = df_valid[df_valid[review_column].str.strip() != ""]
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(df_valid[review_column])
    
    # Hitung cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Identifikasi pasangan yang mirip
    duplicate_pairs = []
    seen_indices = set()
    
    for i in range(len(df_valid)):
        if i in seen_indices:
            continue
            
        # Cari dokumen yang mirip dengan dokumen i
        similar_indices = np.where(cosine_sim[i] > threshold)[0]
        similar_indices = similar_indices[similar_indices != i]  # Hapus self-similarity
        
        if len(similar_indices) > 0:
            group = [i] + list(similar_indices)
            seen_indices.update(group)
            
            # Dapatkan indeks asli dari df
            original_indices = df_valid.iloc[group].index.tolist()
            duplicate_pairs.append(original_indices)
    
    # Hasil
    n_duplicates = sum(len(group) for group in duplicate_pairs)
    n_duplicate_groups = len(duplicate_pairs)
    
    print(f"\nDeteksi Duplikasi (threshold similarity: {threshold}):")
    print(f"- Ditemukan {n_duplicates} ulasan yang kemungkinan duplikat dalam {n_duplicate_groups} kelompok")
    
    if n_duplicate_groups > 0:
        print("\nContoh kelompok duplikat:")
        for i, group in enumerate(duplicate_pairs[:3]):  # Tampilkan 3 kelompok contoh
            print(f"\nKelompok {i+1}:")
            