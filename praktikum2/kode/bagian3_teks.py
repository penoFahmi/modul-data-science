# Import sebaiknya diletakkan di bagian atas file
from collections import Counter
import matplotlib.pyplot as plt

print("=== BAGIAN 3: DATA TAK TERSTRUKTUR (TEKS) ===")

# Baca file teks
with open('komentar.txt', 'r', encoding='utf-8') as file:
    teks = file.read()
print("File teks berhasil dimuat!")

# Info dasar teks
print(f"\nInfo Teks:")
print(f"Panjang karakter: {len(teks)}")
print(f"Jumlah kata: {len(teks.split())}")

# Tampilkan sebagian teks
print(f"\nCuplikan teks (100 karakter pertama):")
print(teks[:100] + "...")

# Pisahkan menjadi kata-kata
kata_kata = teks.lower().split()
print(f"\nJumlah kata setelah dipisah: {len(kata_kata)}")

# Hitung frekuensi kata
frekuensi = Counter(kata_kata)
print("\n5 kata yang paling sering muncul:")
for kata, jumlah in frekuensi.most_common(5):
    print(f"'{kata}': {jumlah} kali")

# Analisis sentimen sederhana
kata_positif = ['bagus', 'baik', 'senang', 'suka', 'hebat']
kata_negatif = ['buruk', 'jelek', 'kecewa', 'marah', 'benci']
jumlah_positif = 0
jumlah_negatif = 0

for kata in kata_kata:
    if kata in kata_positif:
        jumlah_positif += 1
    elif kata in kata_negatif:
        jumlah_negatif += 1

print(f"\nAnalisis Sentimen:")
print(f"Kata positif: {jumlah_positif}")
print(f"Kata negatif: {jumlah_negatif}")

if jumlah_positif > jumlah_negatif:
    sentimen = "Positif"
elif jumlah_negatif > jumlah_positif:
    sentimen = "Negatif"
else:
    sentimen = "Netral"
print(f"Sentimen keseluruhan: {sentimen}")

# Visualisasi kata teratas
kata_teratas = dict(frekuensi.most_common(5))
plt.figure(figsize=(10, 6))
plt.bar(kata_teratas.keys(), kata_teratas.values())
plt.title('5 Kata Paling Sering Muncul')
plt.xlabel('Kata')
plt.ylabel('Frekuensi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("=== SELESAI ===")