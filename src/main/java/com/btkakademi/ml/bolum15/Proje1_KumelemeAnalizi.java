package com.btkakademi.ml.bolum15;

import smile.clustering.DBSCAN;
import smile.clustering.HierarchicalClustering;
import smile.clustering.KMeans;
import smile.clustering.linkage.WardLinkage;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Proje 1: Dünya Ülkeleri Gelişmişlik Kümeleme Analizi
 * Dosya 2/3: Kümeleme Analizi
 * BTK Akademi - Java ile Makine Öğrenmesi
 */
public class Proje1_KumelemeAnalizi {

    private static final int[] IDX = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    private static final String[] ISIM = {
            "GSYIH_Kisi_Basi", "Buyume_Orani", "Issizlik", "Enflasyon",
            "Yasam_Beklentisi", "Bebek_Olum_Hizi", "Okullasma_Orani", "Internet_Erisim",
            "CO2_Emisyon_Kisi", "Saglik_Harcama_GSYIH", "Tarim_GSYIH_Payi", "Sanayi_GSYIH_Payi"
    };

    public static void main(String[] args) {
        try {
            System.out.println("=== PROJE 1: DÜNYA ÜLKELERİ GELİŞMİŞLİK ANALİZİ ===");
            System.out.println("=== DOSYA 2/3: KÜMELEME ANALİZİ ===\n");

            // ============ 1. VERİ YÜKLEME + STANDARDIZASYON ============
            System.out.println("============ 1. VERİ YÜKLEME + STANDARDIZASYON ============\n");

            var is = Proje1_KumelemeAnalizi.class.getClassLoader()
                    .getResourceAsStream("datasets/dunya-ulkeleri-gostergeler.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<String[]> satirlar = new ArrayList<>();
            String satir;
            while ((satir = reader.readLine()) != null) {
                if (!satir.trim().isEmpty()) satirlar.add(satir.split(",", -1));
            }
            reader.close();

            int n = satirlar.size(), m = IDX.length;
            String[] ulkeler = new String[n], bolgeler = new String[n];
            double[][] X = new double[n][m];

            for (int i = 0; i < n; i++) {
                String[] p = satirlar.get(i);
                ulkeler[i] = p[0].trim(); bolgeler[i] = p[1].trim();
                for (int j = 0; j < m; j++) {
                    String d = p[IDX[j]].trim();
                    X[i][j] = (d.isEmpty() || d.equalsIgnoreCase("nan")) ? 0.0 : Double.parseDouble(d);
                }
            }

            // Eksik veri doldurma (0.0 olanları bölge ortalaması ile)
            for (int j = 0; j < m; j++) {
                Map<String, List<Double>> bv = new LinkedHashMap<>();
                double gt = 0; int gs = 0;
                for (int i = 0; i < n; i++) if (X[i][j] != 0) {
                    bv.computeIfAbsent(bolgeler[i], k -> new ArrayList<>()).add(X[i][j]);
                    gt += X[i][j]; gs++;
                }
                double go = gs > 0 ? gt / gs : 0;
                for (int i = 0; i < n; i++) if (X[i][j] == 0) {
                    List<Double> bd = bv.get(bolgeler[i]);
                    X[i][j] = (bd != null && !bd.isEmpty())
                            ? bd.stream().mapToDouble(d -> d).average().orElse(go) : go;
                }
            }

            // Z-Score standardizasyon
            double[] ort = new double[m], std = new double[m];
            for (int j = 0; j < m; j++) { for (int i = 0; i < n; i++) ort[j] += X[i][j]; ort[j] /= n; }
            for (int j = 0; j < m; j++) {
                double tk = 0; for (int i = 0; i < n; i++) tk += (X[i][j] - ort[j]) * (X[i][j] - ort[j]);
                std[j] = Math.sqrt(tk / n);
            }
            double[][] XS = new double[n][m];
            for (int i = 0; i < n; i++) for (int j = 0; j < m; j++)
                XS[i][j] = std[j] > 1e-10 ? (X[i][j] - ort[j]) / std[j] : 0.0;

            System.out.printf("Ülke: %d, Özellik: %d\n", n, m);

            // ============ 2. ELBOW + SILHOUETTE ============
            // Optimal K belirleme: İki yöntem birlikte kullanılır.
            //
            // ELBOW METHOD:
            //   K arttıkça distortion (toplam iç-küme mesafe) düşer.
            //   Başlangıçta hızlı düşer, sonra yavaşlar.
            //   Düşüşün "dirsek" yaptığı K → optimal.
            //   Distortion = Σ (her nokta ile kendi küme merkezinin mesafesi²)
            //
            // SILHOUETTE SKORU:
            //   Her nokta için kümeleme kalitesini ölçer.
            //   s(i) = (b(i) - a(i)) / max(a(i), b(i))
            //     a(i) = kendi kümesindeki diğer noktalara ortalama mesafe
            //     b(i) = en yakın YABaNCI kümeye ortalama mesafe
            //   s = +1 → mükemmel (kendi kümesine çok yakın, diğerine uzak)
            //   s = 0  → sınırda (hangi kümeye ait belirsiz)
            //   s = -1 → yanlış küme (diğer kümeye daha yakın)
            System.out.println("\n============ 2. ELBOW + SILHOUETTE ============\n");

            double enIyiSil = -1; int optK = 2; double oncDist = 0;

            for (int k = 2; k <= 10; k++) {
                var model = KMeans.fit(XS, k, 100); // fit(veri, kümeSayısı, maksİterasyon)
                double dist = model.distortion();    // toplam iç-küme mesafe
                int[] grp = model.group();            // küme atamaları

                // Silhouette hesapla (inline)
                double topSil = 0; int gecerli = 0;
                for (int i = 0; i < n; i++) {
                    double topA = 0; int sayA = 0;
                    for (int j = 0; j < n; j++) if (i != j && grp[j] == grp[i]) {
                        double ms = 0; for (int d = 0; d < m; d++) ms += (XS[i][d] - XS[j][d]) * (XS[i][d] - XS[j][d]);
                        topA += Math.sqrt(ms); sayA++;
                    }
                    double a = sayA > 0 ? topA / sayA : 0, b = Double.MAX_VALUE;
                    for (int c = 0; c < k; c++) if (c != grp[i]) {
                        double topB = 0; int sayB = 0;
                        for (int j = 0; j < n; j++) if (grp[j] == c) {
                            double ms = 0; for (int d = 0; d < m; d++) ms += (XS[i][d] - XS[j][d]) * (XS[i][d] - XS[j][d]);
                            topB += Math.sqrt(ms); sayB++;
                        }
                        if (sayB > 0 && topB / sayB < b) b = topB / sayB;
                    }
                    double mx = Math.max(a, b);
                    if (mx > 1e-10) { topSil += (b - a) / mx; gecerli++; }
                }
                double sil = gecerli > 0 ? topSil / gecerli : 0;

                double deg = k > 2 ? (oncDist - dist) / oncDist * 100 : 0;
                oncDist = dist;
                System.out.printf("K=%-2d  Distortion=%.1f (%%%.0f düşüş)  Silhouette=%.4f%s\n",
                        k, dist, deg, sil, sil > enIyiSil ? " ← En iyi" : "");
                if (sil > enIyiSil) { enIyiSil = sil; optK = k; }
            }

            // ============ 3. K-MEANS ============
            // K-Means algoritması (Lloyd's Algorithm):
            //   1. Rastgele K merkez (centroid) seç
            //   2. Her noktayı en yakın merkeze ata (Öklid mesafesi)
            //   3. Her kümenin yeni merkezini hesapla (üyelerin ortalaması)
            //   4. Atamalar değişmeyene kadar 2-3 tekrarla
            //
            // Smile API: KMeans.fit(data, k, maxIter) → CentroidClustering
            //   .group() → int[] küme atamaları
            //   .distortion() → toplam iç-küme mesafe
            //   .centers() → double[][] küme merkezleri
            //
            // Avantajlar: Basit, hızlı (O(nkm)), büyük veride iyi
            // Dezavantajlar: K önceden bilinmeli, küresel küme varsayar,
            //   başlangıç merkezlere duyarlı (k-means++ bu sorunu hafifletir)
            System.out.println("\n============ 3. K-MEANS ============\n");

            int K = 4; // 4 gelişmişlik düzeyi
            var km = KMeans.fit(XS, K, 100);
            int[] kmGrp = km.group();

            // Küme boyutları + GSYİH ortalaması (etiketleme için)
            int[] boyut = new int[K]; double[] gsyih = new double[K];
            for (int i = 0; i < n; i++) { boyut[kmGrp[i]]++; gsyih[kmGrp[i]] += X[i][0]; }
            for (int k = 0; k < K; k++) gsyih[k] = boyut[k] > 0 ? gsyih[k] / boyut[k] : 0;

            // GSYİH'ye göre sırala → gelişmişlik etiketleri ata
            Integer[] sira = {0, 1, 2, 3};
            Arrays.sort(sira, (a, b) -> Double.compare(gsyih[b], gsyih[a]));
            String[] ETK = {"Yüksek Gelişmiş", "Üst-Orta Gelişmiş", "Alt-Orta Gelişmiş", "Düşük Gelişmiş"};
            String[] etiket = new String[K];
            for (int i = 0; i < K; i++) etiket[sira[i]] = ETK[i];

            for (int k = 0; k < K; k++)
                System.out.printf("Küme %d → %s: %d ülke (GSYİH=$%.0f)\n", k, etiket[k], boyut[k], gsyih[k]);

            // İlk 5 ülke/küme
            System.out.println("\nKüme üyeleri (ilk 5):");
            for (int k = 0; k < K; k++) {
                System.out.printf("  %s: ", etiket[k]);
                int say = 0;
                for (int i = 0; i < n && say < 5; i++)
                    if (kmGrp[i] == k) { if (say > 0) System.out.print(", "); System.out.print(ulkeler[i]); say++; }
                System.out.printf(" ... (%d)\n", boyut[k]);
            }

            // ============ 4. HİYERARŞİK KÜMELEME (WARD) ============
            // Ward Linkage: Birleştirme sırasında küme içi varyansı minimize eder.
            // Agglomerative (aşağıdan yukarı):
            //   Her nokta ayrı küme başlar → en yakın iki küme birleştirilir → tekrarla
            // partition(K): İstenen K sayısına göre dendrogram kesilir.
            //
            // Smile API: WardLinkage.of(data) → HierarchicalClustering.fit(linkage)
            //   .partition(K) → int[] küme atamaları
            //
            // Avantaj: K önceden gerekmez, dendrogram çizilebilir
            // Dezavantaj: O(n²) bellek, büyük veride yavaş
            System.out.println("\n============ 4. HİYERARŞİK (WARD) ============\n");

            var hc = HierarchicalClustering.fit(WardLinkage.of(XS));
            int[] hcGrp = hc.partition(K);
            int[] hcBoy = new int[K];
            for (int a : hcGrp) if (a >= 0 && a < K) hcBoy[a]++;
            for (int k = 0; k < K; k++) System.out.printf("Küme %d: %d ülke\n", k, hcBoy[k]);

            // ============ 5. DBSCAN ============
            // DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
            // Yoğun bölgeleri küme, seyrek bölgeleri noise (gürültü) olarak tanımlar.
            //
            // Parametreler:
            //   minPts: Bir core point olmak için gereken minimum komşu sayısı
            //   eps (ε): Komşuluk yarıçapı (bu mesafe içindeki noktalar komşu)
            //
            // Nokta türleri:
            //   Core point: ε yarıçapında en az minPts komşusu var
            //   Border point: Core point komşusu ama kendisi core değil
            //   Noise: Hiçbir kümeye dahil olamaz → Integer.MAX_VALUE
            //
            // Smile API: DBSCAN.fit(data, minPts, eps)
            //   .group() → küme atamaları (noise = Integer.MAX_VALUE)
            //   .k() → küme sayısı (noise hariç)
            //
            // Avantaj: K gerekmez, farklı şekilli kümeler, noise tespit eder
            // Dezavantaj: eps ve minPts hassas, farklı yoğunlukta kümeler sorunlu
            System.out.println("\n============ 5. DBSCAN ============\n");

            var db = DBSCAN.fit(XS, 5, 3.0); // fit(veri, minPts, eps)
            int[] dbGrp = db.group();
            int noise = 0;
            Map<Integer, Integer> dbBoy = new LinkedHashMap<>();
            for (int a : dbGrp) {
                if (a == Integer.MAX_VALUE) noise++; // noise = MAX_VALUE, -1 DEĞİL!
                else dbBoy.merge(a, 1, Integer::sum);
            }
            System.out.printf("Küme: %d, Noise: %d\n", db.k(), noise);

            // ============ 6. KARŞILAŞTIRMA + TÜRKİYE ============
            System.out.println("\n============ 6. KARŞILAŞTIRMA + TÜRKİYE ============\n");

            // K-Means vs Ward uyumu
            int uyum = 0;
            for (int i = 0; i < n; i++) if (kmGrp[i] == hcGrp[i]) uyum++;
            System.out.printf("K-Means/Ward uyumu: %%%.1f\n", uyum * 100.0 / n);

            // Türkiye
            for (int i = 0; i < n; i++)
                if (ulkeler[i].equalsIgnoreCase("Turkiye")) {
                    System.out.printf("Türkiye K-Means: Küme %d → %s\n", kmGrp[i], etiket[kmGrp[i]]);
                    System.out.printf("Türkiye Ward   : Küme %d\n", hcGrp[i]);
                    System.out.printf("Türkiye DBSCAN : %s\n",
                            dbGrp[i] == Integer.MAX_VALUE ? "Noise" : "Küme " + dbGrp[i]);
                    break;
                }

            System.out.println("\n=== KÜMELEME ANALİZİ TAMAMLANDI ===");

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
