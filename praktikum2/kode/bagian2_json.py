import json

print("=== BAGIAN 2: DATA SEMI-TERSTRUKTUR (JSON) ===")

# Baca file JSON
with open('profil.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print("Data JSON berhasil dimuat!")

# Akses data langsung
print(f"\nNama: {data['nama']}")
print(f"Umur: {data['umur']}")
print(f"Jurusan: {data['jurusan']}")

# Akses data nested (alamat)
print(f"\nAlamat:")
print(f"Jalan: {data['alamat']['jalan']}")
print(f"Kota: {data['alamat']['kota']}")
print(f"Kode Pos: {data['alamat']['kode_pos']}")

# Akses data array (hobi)
print(f"\nHobi:")
for i, hobi in enumerate(data['hobi'], 1):
    print(f"{i}. {hobi}")

# Akses data nested dalam array (nilai)
print(f"\nNilai Mata Kuliah:")
for matkul in data['nilai']:
    print(f"- {matkul['nama']}: {matkul['skor']}")

# Hitung rata-rata nilai
total_nilai = sum([matkul['skor'] for matkul in data['nilai']])
rata_nilai = total_nilai / len(data['nilai'])
print(f"\nRata-rata nilai: {rata_nilai:.2f}")

# Tampilkan struktur JSON dengan indentasi
print(f"\nStruktur JSON lengkap:")
print(json.dumps(data, indent=2, ensure_ascii=False))

print("=== SELESAI ===")