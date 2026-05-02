import pandas as pd

# Load kedua dataset
df_bni = pd.read_csv('bank_Jateng3.csv')
df_jateng = pd.read_csv('bank_Jateng_merged.csv')

# Standarisasi nama kolom (jika perlu) agar bisa digabung
df_bni.columns = df_bni.columns.str.strip()
df_jateng.columns = df_jateng.columns.str.strip()

# Cek apakah kolom cocok
if not df_bni.columns.equals(df_jateng.columns):
    # Jika tidak cocok, tampilkan kolom untuk dicek manual
    print("Kolom tidak sama:")
    print("Jateng2:", df_bni.columns)
    print("Jateng:", df_jateng.columns)
else:
    # Gabungkan dua dataframe
    df_combined = pd.concat([df_bni, df_jateng], ignore_index=True)

    # Hapus duplikat jika ada
    df_combined = df_combined.drop_duplicates()

    # Simpan hasil gabungan ke file baru
    df_combined.to_csv('bank_Jateng.csv', index=False)
    print("Dataset berhasil digabung dan disimpan sebagai bank_Jateng.csv")
