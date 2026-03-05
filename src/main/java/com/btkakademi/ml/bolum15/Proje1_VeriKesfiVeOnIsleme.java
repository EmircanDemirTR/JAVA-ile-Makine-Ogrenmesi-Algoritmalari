package com.btkakademi.ml.bolum15;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Proje 1: Dünya Ülkeleri Gelişmişlik Kümeleme Analizi
 * Dosya 1/3: Veri Keşfi ve Ön İşleme
 * BTK Akademi - Java ile Makine Öğrenmesi
 */
public class Proje1_VeriKesfiVeOnIsleme {

    // CSV sütun yapısı (0-bazlı):
    // Ulke(0), Bolge(1), Gelir_Grubu(2), Nufus(3), GSYIH_Kisi_Basi(4),
    // Buyume_Orani(5), Issizlik(6), Enflasyon(7), Yasam_Beklentisi(8),
    // Bebek_Olum_Hizi(9), Okullasma_Orani(10), Internet_Erisim(11),
    // CO2_Emisyon_Kisi(12), Saglik_Harcama_GSYIH(13), Tarim_GSYIH_Payi(14), Sanayi_GSYIH_Payi(15)

    // Kümeleme özellikleri — Nufus hariç (ölçeği çok farklı, kümelemeyi bozar)
    private static final int[] IDX = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    private static final String[] ISIM = {
            "GSYIH_Kisi_Basi", "Buyume_Orani", "Issizlik", "Enflasyon",
            "Yasam_Beklentisi", "Bebek_Olum_Hizi", "Okullasma_Orani", "Internet_Erisim",
            "CO2_Emisyon_Kisi", "Saglik_Harcama_GSYIH", "Tarim_GSYIH_Payi", "Sanayi_GSYIH_Payi"
    };

    public static void main(String[] args) {
        try {
            System.out.println("=== PROJE 1: DÜNYA ÜLKELERİ GELİŞMİŞLİK ANALİZİ ===");
            System.out.println("=== DOSYA 1/3: VERİ KEŞFİ VE ÖN İŞLEME ===\n");

            // ============ 1. VERİ YÜKLEME ============
            // CSV (Comma-Separated Values): En yaygın tablo formatı.
            // Her satır bir gözlem (ülke), virgülle ayrılmış sütunlar özellikler.
            // İlk satır "header" (başlık) → sütun isimlerini içerir, veri değil.
            // split(",", -1): Virgülle ayır — -1 parametresi boş alanları korur.
            // Boş veya "nan" alanlar → Double.NaN (Not a Number) olarak işaretlenir.
            // NaN, Java'da "tanımsız değer" demek — hesaplamalara katılmaz.
            System.out.println("============ 1. VERİ YÜKLEME ============\n");

            var is = Proje1_VeriKesfiVeOnIsleme.class.getClassLoader()
                    .getResourceAsStream("datasets/dunya-ulkeleri-gostergeler.csv"); // classpath'ten oku
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine(); // header satırını atla

            List<String[]> satirlar = new ArrayList<>();
            String satir;
            while ((satir = reader.readLine()) != null) {
                if (!satir.trim().isEmpty()) satirlar.add(satir.split(",", -1));
            }
            reader.close();

            int n = satirlar.size(), m = IDX.length; // n=ülke sayısı, m=özellik sayısı
            String[] ulkeler = new String[n], bolgeler = new String[n];
            double[][] X = new double[n][m]; // kümeleme veri matrisi

            for (int i = 0; i < n; i++) {
                String[] p = satirlar.get(i);
                ulkeler[i] = p[0].trim();  // ülke adı
                bolgeler[i] = p[1].trim(); // coğrafi bölge
                for (int j = 0; j < m; j++) {
                    String d = p[IDX[j]].trim();
                    X[i][j] = (d.isEmpty() || d.equalsIgnoreCase("nan")) ? Double.NaN : Double.parseDouble(d);
                }
            }
            System.out.printf("Ülke: %d, Özellik: %d\n", n, m);

            // ============ 2. EKSİK VERİ DOLDURMA ============
            // Eksik veri (missing data): Ölçülmemiş, kayıp veya hatalı değerler.
            // Makine öğrenmesi algoritmaları NaN ile çalışamaz → doldurulmalı.
            //
            // Doldurma stratejileri:
            //   Ortalama: Basit ama aykırılara duyarlı
            //   Medyan: Aykırılara dayanıklı (robust)
            //   Bölge ortalaması: Benzer ülkelerin değerleri kullanılır (akıllı)
            //   KNN-Imputation: En yakın K komşunun ortalaması (gelişmiş)
            //
            // Biz bölge ortalaması kullanıyoruz — aynı bölgedeki ülkeler
            // benzer ekonomik ve sosyal göstergelere sahip olur.
            // Bölgede hiç veri yoksa → genel ortalama (fallback).
            System.out.println("\n============ 2. EKSİK VERİ DOLDURMA ============\n");

            int doldurulan = 0;
            for (int j = 0; j < m; j++) {
                Map<String, List<Double>> bolgeVeri = new LinkedHashMap<>();
                double genTop = 0; int genSay = 0;
                for (int i = 0; i < n; i++) {
                    if (!Double.isNaN(X[i][j])) {
                        bolgeVeri.computeIfAbsent(bolgeler[i], k -> new ArrayList<>()).add(X[i][j]);
                        genTop += X[i][j]; genSay++;
                    }
                }
                double genOrt = genSay > 0 ? genTop / genSay : 0;
                for (int i = 0; i < n; i++) {
                    if (Double.isNaN(X[i][j])) {
                        List<Double> bd = bolgeVeri.get(bolgeler[i]);
                        X[i][j] = (bd != null && !bd.isEmpty())
                                ? bd.stream().mapToDouble(d -> d).average().orElse(genOrt) : genOrt;
                        doldurulan++;
                    }
                }
            }
            System.out.printf("Doldurulan: %d hücre\n", doldurulan);

            // ============ 3. AYKIRI DEĞER TESPİTİ (IQR) ============
            // IQR (Interquartile Range) — Çeyrekler Arası Genişlik:
            // Verinin orta %50'sinin yayılımını ölçer.
            //
            // Q1 = 25. yüzdelik (verinin %25'i bunun altında)
            // Q3 = 75. yüzdelik (verinin %75'i bunun altında)
            // IQR = Q3 - Q1 (ortadaki kutunun yüksekliği)
            //
            // Aykırı değer sınırları (Tukey kuralı):
            //   Alt sınır = Q1 - 1.5 × IQR
            //   Üst sınır = Q3 + 1.5 × IQR
            //   Bu sınırların dışı → outlier (aykırı)
            //
            // Kümelemede aykırılar merkezleri çarpıtır.
            // Standardizasyon etkiyi azaltır ama tamamen çözmez.
            // Bazı projelerde aykırılar çıkarılır (winsorization).
            System.out.println("\n============ 3. AYKIRI DEĞER TESPİTİ (IQR) ============\n");

            for (int j = 0; j < m; j++) {
                double[] s = new double[n];
                for (int i = 0; i < n; i++) s[i] = X[i][j];
                Arrays.sort(s);
                double q1 = s[(int) (n * 0.25)], q3 = s[(int) (n * 0.75)], iqr = q3 - q1;
                int aykiri = 0;
                for (int i = 0; i < n; i++)
                    if (X[i][j] < q1 - 1.5 * iqr || X[i][j] > q3 + 1.5 * iqr) aykiri++;
                if (aykiri > 0) System.out.printf("%-22s: %d aykırı\n", ISIM[j], aykiri);
            }

            // ============ 4. KORELASYON ANALİZİ ============
            // Pearson korelasyon katsayısı (r): Doğrusal ilişkinin gücü ve yönü.
            // r = Σ((xi-x̄)(yi-ȳ)) / √(Σ(xi-x̄)² × Σ(yi-ȳ)²)
            //
            //   r = +1  → mükemmel pozitif (biri artınca diğeri de artar)
            //   r = -1  → mükemmel negatif (biri artınca diğeri azalır)
            //   r = 0   → doğrusal ilişki yok
            //   |r| > 0.7 → güçlü korelasyon
            //   |r| < 0.3 → zayıf korelasyon
            //
            // Örnek: GSYİH ↑ → İnternet Erişimi ↑ (pozitif korelasyon)
            //         GSYİH ↑ → Bebek Ölüm Hızı ↓ (negatif korelasyon)
            //
            // Yüksek korelasyonlu özellikler benzer bilgi taşır → PCA ile birleştirilebilir.
            System.out.println("\n============ 4. KORELASYON ANALİZİ ============\n");

            double[] ort = new double[m];
            for (int j = 0; j < m; j++) { for (int i = 0; i < n; i++) ort[j] += X[i][j]; ort[j] /= n; }

            for (int a = 0; a < m; a++) for (int b = a + 1; b < m; b++) {
                double ab = 0, a2 = 0, b2 = 0;
                for (int i = 0; i < n; i++) {
                    double fa = X[i][a] - ort[a], fb = X[i][b] - ort[b];
                    ab += fa * fb; a2 += fa * fa; b2 += fb * fb;
                }
                double r = Math.sqrt(a2 * b2) > 1e-10 ? ab / Math.sqrt(a2 * b2) : 0;
                if (Math.abs(r) > 0.7) System.out.printf("  %s - %s: r=%.4f\n", ISIM[a], ISIM[b], r);
            }

            // ============ 5. Z-SCORE STANDARDIZASYON ============
            // Formül: z = (x - μ) / σ   (μ = ortalama, σ = standart sapma)
            // Sonuç: Her özellik → ortalama=0, standart sapma=1
            //
            // Neden zorunlu?
            //   GSYİH: 0 – 100.000$  vs  İşsizlik: 0 – 30%
            //   Ölçekler çok farklı → mesafe hesabında GSYİH baskın olur.
            //   K-Means Öklid mesafesi kullanır → büyük ölçekli özellik domine eder.
            //   Standardizasyon sonrası tüm özellikler eşit ağırlıkta.
            //
            // PCA için de zorunlu — varyans bazlı çalışır,
            // standardize edilmezse yüksek ölçekli özellik tüm varyansı alır.
            //
            // Alternatif: Min-Max → [0,1] aralığı. KNN için daha uygun olabilir.
            System.out.println("\n============ 5. Z-SCORE STANDARDIZASYON ============\n");

            double[] std = new double[m];
            for (int j = 0; j < m; j++) {
                double topK = 0;
                for (int i = 0; i < n; i++) topK += (X[i][j] - ort[j]) * (X[i][j] - ort[j]);
                std[j] = Math.sqrt(topK / n);
            }
            double[][] XStd = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    XStd[i][j] = std[j] > 1e-10 ? (X[i][j] - ort[j]) / std[j] : 0.0;

            System.out.println("Standardizasyon tamamlandı.");

            // ============ 6. TÜRKİYE ============
            // Z > 0 → dünya ortalamasının üstünde
            // Z < 0 → dünya ortalamasının altında
            // |Z| > 1 → belirgin fark, |Z| > 2 → çok belirgin
            System.out.println("\n============ 6. TÜRKİYE ============\n");

            for (int i = 0; i < n; i++)
                if (ulkeler[i].equalsIgnoreCase("Turkiye")) {
                    for (int j = 0; j < m; j++)
                        System.out.printf("  %-22s: Ham=%.1f, Z=%.3f\n", ISIM[j], X[i][j], XStd[i][j]);
                    break;
                }

            System.out.println("\n=== VERİ KEŞFİ VE ÖN İŞLEME TAMAMLANDI ===");

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
