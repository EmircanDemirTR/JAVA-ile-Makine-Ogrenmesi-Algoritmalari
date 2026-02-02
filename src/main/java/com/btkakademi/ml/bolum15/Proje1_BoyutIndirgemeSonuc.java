package com.btkakademi.ml.bolum15;

import smile.clustering.KMeans;
import smile.feature.extraction.PCA;
import smile.manifold.TSNE;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Proje 1: Dünya Ülkeleri Gelişmişlik Kümeleme Analizi
 * Dosya 3/3: Boyut İndirgeme ve Sonuç Raporu
 * BTK Akademi - Java ile Makine Öğrenmesi
 */
public class Proje1_BoyutIndirgemeSonuc {

    private static final int[] IDX = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    private static final String[] ISIM = {
            "GSYIH_Kisi_Basi", "Buyume_Orani", "Issizlik", "Enflasyon",
            "Yasam_Beklentisi", "Bebek_Olum_Hizi", "Okullasma_Orani", "Internet_Erisim",
            "CO2_Emisyon_Kisi", "Saglik_Harcama_GSYIH", "Tarim_GSYIH_Payi", "Sanayi_GSYIH_Payi"
    };

    public static void main(String[] args) {
        try {
            System.out.println("=== PROJE 1: DÜNYA ÜLKELERİ GELİŞMİŞLİK ANALİZİ ===");
            System.out.println("=== DOSYA 3/3: BOYUT İNDİRGEME VE SONUÇ ===\n");

            // ============ 1. VERİ HAZIRLIK ============
            System.out.println("============ 1. VERİ HAZIRLIK ============\n");

            var is = Proje1_BoyutIndirgemeSonuc.class.getClassLoader()
                    .getResourceAsStream("datasets/dunya-ulkeleri-gostergeler.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();
            List<String[]> satirlar = new ArrayList<>();
            String satir;
            while ((satir = reader.readLine()) != null)
                if (!satir.trim().isEmpty()) satirlar.add(satir.split(",", -1));
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

            // Eksik doldur + Z-Score (inline)
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
            double[] ort = new double[m], std = new double[m];
            for (int j = 0; j < m; j++) { for (int i = 0; i < n; i++) ort[j] += X[i][j]; ort[j] /= n; }
            for (int j = 0; j < m; j++) {
                double tk = 0; for (int i = 0; i < n; i++) tk += (X[i][j] - ort[j]) * (X[i][j] - ort[j]);
                std[j] = Math.sqrt(tk / n);
            }
            double[][] XS = new double[n][m];
            for (int i = 0; i < n; i++) for (int j = 0; j < m; j++)
                XS[i][j] = std[j] > 1e-10 ? (X[i][j] - ort[j]) / std[j] : 0.0;

            // K-Means (küme etiketleri ile gösterim için)
            int K = 4;
            int[] grp = KMeans.fit(XS, K, 100).group();
            double[] gsyih = new double[K]; int[] boy = new int[K];
            for (int i = 0; i < n; i++) { gsyih[grp[i]] += X[i][0]; boy[grp[i]]++; }
            for (int k = 0; k < K; k++) gsyih[k] = boy[k] > 0 ? gsyih[k] / boy[k] : 0;
            Integer[] sr = {0, 1, 2, 3};
            Arrays.sort(sr, (a, b) -> Double.compare(gsyih[b], gsyih[a]));
            String[] ETK = {"Yüksek", "Üst-Orta", "Alt-Orta", "Düşük"};
            String[] et = new String[K];
            for (int i = 0; i < K; i++) et[sr[i]] = ETK[i];

            System.out.printf("Veri: %d ülke × %d özellik, %d küme\n", n, m, K);

            // ============ 2. PCA ANALİZİ ============
            // PCA (Principal Component Analysis — Temel Bileşen Analizi):
            // Yüksek boyutlu veriyi daha az boyuta indirger.
            // Kovaryans matrisinin özvektörlerini (eigenvectors) bulur.
            // Her "temel bileşen" (PC) bir özvektör yönünde maksimum varyansı yakalar.
            //
            // PC1: En fazla varyansı açıklayan yön (genellikle "gelişmişlik ekseni")
            // PC2: PC1'e dik, ikinci en çok varyans
            // Her PC öncekine diktir (ortogonal) → bilgi tekrarı yok.
            //
            // Varyans açıklama oranı: PC1 ne kadar bilgi taşıyor?
            // Kümülatif %90 → o kadar PC yeterli, geri kalanı "gürültü".
            //
            // Smile API:
            //   PCA.fit(XStd) → model
            //   .varianceProportion() → her PC'nin varyans oranı
            //   .cumulativeVarianceProportion() → kümülatif oran
            //   .getProjection(k).apply(X) → k boyuta projeksiyon
            //
            // DİKKAT: Standardizasyon ZORUNLU! Aksi halde yüksek ölçekli
            // özellik tüm varyansı alır, PCA anlamsızlaşır.
            System.out.println("\n============ 2. PCA ANALİZİ ============\n");

            var pca = PCA.fit(XS);
            // Smile 5.1.0: toArray() parametresiz çalışmaz — boyut belirtmeli
            var vv = pca.varianceProportion();
            double[] vp = vv.toArray(new double[vv.size()]);
            var kv = pca.cumulativeVarianceProportion();
            double[] kp = kv.toArray(new double[kv.size()]);

            for (int i = 0; i < m; i++)
                System.out.printf("PC%-2d: %%%.2f (Küm: %%%.2f)\n", i + 1, vp[i] * 100, kp[i] * 100);

            int pc90 = 0;
            for (int i = 0; i < kp.length; i++) if (kp[i] >= 0.90) { pc90 = i + 1; break; }
            System.out.printf("\n%%90 varyans → %d bileşen\n", pc90);

            // ============ 3. PCA 2D PROJEKSİYON ============
            // 12 boyut → 2 boyut: Görselleştirme için.
            // getProjection(2) → ilk 2 PC seçilir
            // apply(XStd) → her ülke 2D'ye projekte edilir
            System.out.println("\n============ 3. PCA 2D PROJEKSİYON ============\n");

            double[][] X2d = pca.getProjection(2).apply(XS);
            System.out.printf("PC1(%%%.1f) + PC2(%%%.1f) = %%%.1f\n", vp[0] * 100, vp[1] * 100, kp[1] * 100);

            String[] secili = {"Turkiye", "ABD", "Almanya", "Cin", "Hindistan", "Japonya", "Norvec", "Nijerya"};
            for (String u : secili)
                for (int i = 0; i < n; i++) if (ulkeler[i].equalsIgnoreCase(u)) {
                    System.out.printf("  %-12s: PC1=%.3f PC2=%.3f → %s\n", ulkeler[i], X2d[i][0], X2d[i][1], et[grp[i]]);
                    break;
                }

            // ============ 4. t-SNE ============
            // t-SNE (t-distributed Stochastic Neighbor Embedding):
            // Yüksek boyutlu veriyi 2D'ye indirger — sadece GÖRSELLEŞTİRME amaçlı.
            //
            // Çalışma prensibi:
            //   1. Yüksek boyutta her nokta çifti arası "benzerlik" hesapla (Gaussian)
            //   2. Düşük boyutta aynı benzerlikleri yeniden oluşturmaya çalış (t-dağılımı)
            //   3. KL-divergence minimize et (iteratif optimizasyon)
            //
            // Parametreler:
            //   perplexity (5-50): Komşuluk ölçeği
            //     Küçük → yerel yapı ön plana çıkar
            //     Büyük → küresel yapı daha belirgin
            //   eta (öğrenme hızı): 10-1000, genellikle 200
            //   maxIter: İterasyon sayısı, 1000 genellikle yeter
            //
            // PCA vs t-SNE:
            //   PCA: Doğrusal, deterministik, hızlı, varyans bilgisi var
            //   t-SNE: Doğrusal olmayan, rastgele, yavaş, varyans yok
            //   PCA: Küresel yapıyı korur — t-SNE: Yerel yapıyı korur
            //   PCA: Yeni veri dönüştürülebilir — t-SNE: Sadece fit zamanında
            System.out.println("\n============ 4. t-SNE ============\n");

            // Options(boyut, perplexity, eta, earlyExaggeration, maxIter)
            var tsne = TSNE.fit(XS, new TSNE.Options(2, 30.0, 200.0, 12.0, 1000));
            double[][] T = tsne.coordinates();

            System.out.println("t-SNE tamamlandı (perplexity=30, eta=200)");
            for (String u : secili)
                for (int i = 0; i < n; i++) if (ulkeler[i].equalsIgnoreCase(u)) {
                    System.out.printf("  %-12s: t1=%.3f t2=%.3f → %s\n", ulkeler[i], T[i][0], T[i][1], et[grp[i]]);
                    break;
                }

            // ============ 5. LOADINGS ============
            // Loading (bileşen yükü) = orijinal özellik ile PC skoru arasındaki korelasyon.
            // Yüksek |loading| → o özellik bu PC'yi güçlü etkiliyor.
            // PC1'de pozitif loading: GSYİH, yaşam beklentisi, internet → "gelişmişlik"
            // PC1'de negatif loading: Bebek ölüm, tarım payı → "az gelişmişlik"
            System.out.println("\n============ 5. LOADINGS ============\n");

            for (int j = 0; j < m; j++) {
                // Pearson korelasyonu (inline): ozellik vs PC1
                double ox = 0, oy = 0;
                for (int i = 0; i < n; i++) { ox += XS[i][j]; oy += X2d[i][0]; }
                ox /= n; oy /= n;
                double xy = 0, x2 = 0, y2 = 0;
                for (int i = 0; i < n; i++) {
                    double fx = XS[i][j] - ox, fy = X2d[i][0] - oy;
                    xy += fx * fy; x2 += fx * fx; y2 += fy * fy;
                }
                double r = Math.sqrt(x2 * y2) > 1e-10 ? xy / Math.sqrt(x2 * y2) : 0;
                System.out.printf("  %-22s: PC1=%.4f\n", ISIM[j], r);
            }

            // ============ 6. SONUÇ ============
            System.out.println("\n============ 6. SONUÇ ============\n");

            for (int i = 0; i < K; i++)
                System.out.printf("%s: %d ülke (GSYİH=$%.0f)\n", ETK[i], boy[sr[i]], gsyih[sr[i]]);
            System.out.printf("PCA: %d PC ile %%90 varyans\n", pc90);

            System.out.println("\n=== PROJE 1 TAMAMLANDI ===");

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
