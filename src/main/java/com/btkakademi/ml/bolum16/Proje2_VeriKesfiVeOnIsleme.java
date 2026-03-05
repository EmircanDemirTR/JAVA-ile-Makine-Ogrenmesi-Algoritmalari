package com.btkakademi.ml.bolum16;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Proje 2: Kalp Hastalığı Risk Tahmin Sistemi
 * Dosya 1/3: Veri Keşfi ve Ön İşleme
 * Veri Seti: Heart Failure Prediction (Kaggle/UCI)
 * BTK Akademi - Java ile Makine Öğrenmesi
 */
public class Proje2_VeriKesfiVeOnIsleme {

    public static void main(String[] args) {
        try {
            System.out.println("=== PROJE 2: KALP HASTALIĞI RİSK TAHMİN SİSTEMİ ===");
            System.out.println("=== DOSYA 1/3: VERİ KEŞFİ VE ÖN İŞLEME ===\n");

            // ============ 1. VERİ YÜKLEME ============
            System.out.println("============ 1. VERİ YÜKLEME ============\n");

            var is = Proje2_VeriKesfiVeOnIsleme.class.getClassLoader()
                    .getResourceAsStream("datasets/kalp-hastaligi.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            String[] basliklar = reader.readLine().split(",");

            List<String[]> satirlar = new ArrayList<>();
            String satir;
            while ((satir = reader.readLine()) != null)
                if (!satir.trim().isEmpty()) satirlar.add(satir.split(",", -1));
            reader.close();

            int n = satirlar.size();
            Map<String, Integer> col = new LinkedHashMap<>(); // sütun adı → indeks
            for (int i = 0; i < basliklar.length; i++) col.put(basliklar[i].trim(), i);

            System.out.printf("Kayıt: %d, Sütun: %d\n", n, basliklar.length);

            // ============ 2. EDA (KEŞİFSEL VERİ ANALİZİ) ============
            // EDA amacı: Veri setini tanımak — dağılım, eksiklik, dengesizlik.
            // İlk adım daima sınıf dağılımına bakmaktır:
            //   Dengeli mi? → Ek işlem gerekmez
            //   Dengesiz mi? → SMOTE, class weight, undersampling gerekebilir
            //   Oran < 1.5 → dengeli kabul edilir
            System.out.println("\n============ 2. EDA ============\n");

            int hIdx = basliklar.length - 1; // hedef sütun (son sütun)
            int[] sinif = new int[2]; // 0=sağlıklı, 1=hasta
            for (String[] p : satirlar) sinif[Integer.parseInt(p[hIdx].trim())]++;

            double oran = (double) Math.max(sinif[0], sinif[1]) / Math.min(sinif[0], sinif[1]);
            System.out.printf("Sağlıklı: %d, Hasta: %d, Oran: %.2f:1 (%s)\n",
                    sinif[0], sinif[1], oran, oran < 1.5 ? "dengeli" : "dengesiz");

            // ============ 3. EKSİK VERİ DOLDURMA ============
            // Kolesterol ve Kan Basıncı'nda 0 değerleri klinik olarak imkansız → eksik.
            // Neden medyan? Ortalama aykırı değerlerden etkilenir.
            //   Örnek: [100, 120, 130, 900] → Ort=312.5 (yanıltıcı), Medyan=125 (gerçekçi)
            // Medyan = Sıralı verinin tam ortasındaki değer.
            //   Çift elemanda: ortadaki iki değerin ortalaması.
            System.out.println("\n============ 3. EKSİK VERİ DOLDURMA ============\n");

            int kolIdx = col.get("Kolesterol"), kbIdx = col.get("Dinlenme_Kan_Basinci");

            // Medyanları hesapla (sıfır olmayanlardan)
            List<Double> kolG = new ArrayList<>(), kbG = new ArrayList<>();
            for (String[] p : satirlar) {
                double kol = Double.parseDouble(p[kolIdx].trim());
                double kb = Double.parseDouble(p[kbIdx].trim());
                if (kol > 0) kolG.add(kol);
                if (kb > 0) kbG.add(kb);
            }
            Collections.sort(kolG); Collections.sort(kbG);
            // Medyan inline: ortadaki değer(ler)in ortalaması
            double kolMed = kolG.size() % 2 == 0
                    ? (kolG.get(kolG.size() / 2 - 1) + kolG.get(kolG.size() / 2)) / 2.0 : kolG.get(kolG.size() / 2);
            double kbMed = kbG.size() % 2 == 0
                    ? (kbG.get(kbG.size() / 2 - 1) + kbG.get(kbG.size() / 2)) / 2.0 : kbG.get(kbG.size() / 2);

            int kolEk = 0, kbEk = 0;
            for (String[] p : satirlar) {
                if (Double.parseDouble(p[kolIdx].trim()) == 0) { p[kolIdx] = String.valueOf((int) kolMed); kolEk++; }
                if (Double.parseDouble(p[kbIdx].trim()) == 0) { p[kbIdx] = String.valueOf((int) kbMed); kbEk++; }
            }
            System.out.printf("Kolesterol: %d eksik → medyan=%.0f\nKan Basıncı: %d eksik → medyan=%.0f\n",
                    kolEk, kolMed, kbEk, kbMed);

            // ============ 4. KATEGORİK ENCODING ============
            // Makine öğrenmesi algoritmaları sayısal giriş ister.
            // Kategorik (metin) veriler sayıya dönüştürülmeli.
            //
            // One-Hot Encoding: Her kategori ayrı sütun olur (0/1).
            //   Göğüs ağrısı: TA, ATA, NAP, ASY → 4 sütun
            //   Neden? Kategoriler arası sıra yoksa ordinal yanıltıcı olur.
            //   Örnek: TA=1, ATA=2 desek → "ATA > TA" gibi yanlış ilişki kurulur.
            //
            // Ordinal Encoding: Kategorilere sıralı sayı verilir.
            //   ST Eğim: Aşağı=0 < Düz=1 < Yukarı=2 → doğal sıra var.
            //   Cinsiyet: E=1, K=0 → binary, sıra yok ama 2 kategoride ordinal OK.
            //
            // Binary: Zaten 0/1 olan veriler (Açlık Kan Şekeri).
            //
            // KURAL: Sırasız kategorilere ordinal uygulamak MODEL HATASI yaratır!
            System.out.println("\n============ 4. KATEGORİK ENCODING ============\n");

            // 16 özellik: Yas, Cinsiyet, TA, ATA, NAP, ASY, DKB, Kolesterol,
            //   AKS, EKG_Normal, EKG_ST, EKG_LVH, MKH, Angina, Oldpeak, ST_Egim
            int F = 16;
            double[][] XE = new double[n][F];
            int[] y = new int[n];
            String[] FN = {"Yas", "Cinsiyet", "Gogus_TA", "Gogus_ATA", "Gogus_NAP", "Gogus_ASY",
                    "DKB", "Kolesterol", "AKS", "EKG_Normal", "EKG_ST", "EKG_LVH",
                    "MKH", "Angina", "Oldpeak", "ST_Egim"};

            for (int i = 0; i < n; i++) {
                String[] p = satirlar.get(i);
                XE[i][0] = Double.parseDouble(p[col.get("Yas")].trim());
                XE[i][1] = p[col.get("Cinsiyet")].trim().equals("E") ? 1 : 0;
                String g = p[col.get("Gogus_Agrisi_Tipi")].trim();
                XE[i][2] = g.equals("TA") ? 1 : 0; XE[i][3] = g.equals("ATA") ? 1 : 0;
                XE[i][4] = g.equals("NAP") ? 1 : 0; XE[i][5] = g.equals("ASY") ? 1 : 0;
                XE[i][6] = Double.parseDouble(p[kbIdx].trim());
                XE[i][7] = Double.parseDouble(p[kolIdx].trim());
                XE[i][8] = Double.parseDouble(p[col.get("Aclik_Kan_Sekeri")].trim());
                String e = p[col.get("Dinlenme_EKG")].trim();
                XE[i][9] = e.equals("Normal") ? 1 : 0; XE[i][10] = e.equals("ST") ? 1 : 0;
                XE[i][11] = e.equals("LVH") ? 1 : 0;
                XE[i][12] = Double.parseDouble(p[col.get("Maks_Kalp_Hizi")].trim());
                XE[i][13] = p[col.get("Egzersiz_Angina")].trim().equals("E") ? 1 : 0;
                XE[i][14] = Double.parseDouble(p[col.get("ST_Oldpeak")].trim());
                String eg = p[col.get("ST_Egim")].trim();
                XE[i][15] = eg.equals("Asagi") ? 0 : eg.equals("Duz") ? 1 : 2;
                y[i] = Integer.parseInt(p[hIdx].trim());
            }
            System.out.printf("Encoding sonrası: %d özellik\n", F);

            // ============ 5. MIN-MAX NORMALİZASYON ============
            // Formül: x_norm = (x - min) / (max - min) → [0, 1] aralığı
            // Neden? KNN ve SVM mesafe tabanlı — farklı ölçekler mesafeyi bozar.
            //   Yaş: 28-77 vs AKS: 0-1 → normalize edilmezse yaş baskın olur.
            // Z-Score da kullanılabilir (ort=0, std=1) ama [0,1] daha sezgisel.
            System.out.println("\n============ 5. MIN-MAX NORMALİZASYON ============\n");

            double[] mn = new double[F], mx = new double[F];
            Arrays.fill(mn, Double.MAX_VALUE); Arrays.fill(mx, -Double.MAX_VALUE);
            for (double[] r : XE) for (int j = 0; j < F; j++) {
                if (r[j] < mn[j]) mn[j] = r[j]; if (r[j] > mx[j]) mx[j] = r[j];
            }
            double[][] XN = new double[n][F];
            for (int i = 0; i < n; i++) for (int j = 0; j < F; j++)
                XN[i][j] = (XE[i][j] - mn[j]) / (mx[j] - mn[j] + 1e-10);

            System.out.println("Normalizasyon tamamlandı → [0, 1]");

            // ============ 6. STRATIFIED TRAIN/TEST SPLIT ============
            // Stratified: Her sınıfın oranı train ve test'te korunur.
            //   Toplam %55 hasta ise → train'de de ~%55, test'te de ~%55
            //   Neden? Rastgele split'te azınlık sınıfı test'te yok olabilir.
            //
            // İşlem:
            //   1. Sınıf 0 ve Sınıf 1 indeksleri ayrı listelere al
            //   2. Her listeyi karıştır (shuffle)
            //   3. Her listeden %70'i eğitime, %30'u teste ayır
            //   4. Birleştir ve tekrar karıştır
            //
            // Seed (42): Rastgelelik tohumu — aynı sonuçlar tekrarlanır.
            System.out.println("\n============ 6. STRATIFIED TRAIN/TEST SPLIT ============\n");

            List<Integer> s0 = new ArrayList<>(), s1 = new ArrayList<>();
            for (int i = 0; i < n; i++) { if (y[i] == 0) s0.add(i); else s1.add(i); }
            Random rnd = new Random(42);
            Collections.shuffle(s0, rnd); Collections.shuffle(s1, rnd);
            int e0 = (int) (s0.size() * 0.7), e1 = (int) (s1.size() * 0.7);

            List<Integer> egIdx = new ArrayList<>(), teIdx = new ArrayList<>();
            egIdx.addAll(s0.subList(0, e0)); egIdx.addAll(s1.subList(0, e1));
            teIdx.addAll(s0.subList(e0, s0.size())); teIdx.addAll(s1.subList(e1, s1.size()));
            Collections.shuffle(egIdx, rnd); Collections.shuffle(teIdx, rnd);

            System.out.printf("Eğitim: %d, Test: %d\n", egIdx.size(), teIdx.size());

            // ============ 7. KORELASYON ANALİZİ ============
            // Hedef ile özellik arası Pearson korelasyonu.
            // Pozitif r → özellik arttıkça hastalık riski artar (Oldpeak, Angina)
            // Negatif r → özellik arttıkça risk azalır (MKH — kalp hızı iyi gösterge)
            // |r| > 0.3 → güçlü → sınıflandırmada etkili özellik
            // |r| < 0.1 → zayıf → modele katkısı az
            System.out.println("\n============ 7. KORELASYON ANALİZİ ============\n");

            double[] yD = new double[n];
            for (int i = 0; i < n; i++) yD[i] = y[i];

            double[][] korArr = new double[F][2]; // [r, |r|]
            for (int j = 0; j < F; j++) {
                // Pearson inline
                double ox = 0, oy = 0;
                for (int i = 0; i < n; i++) { ox += XN[i][j]; oy += yD[i]; }
                ox /= n; oy /= n;
                double xy = 0, x2 = 0, y2 = 0;
                for (int i = 0; i < n; i++) {
                    double fx = XN[i][j] - ox, fy = yD[i] - oy;
                    xy += fx * fy; x2 += fx * fx; y2 += fy * fy;
                }
                double r = Math.sqrt(x2 * y2) > 1e-10 ? xy / Math.sqrt(x2 * y2) : 0;
                korArr[j][0] = r; korArr[j][1] = Math.abs(r);
            }
            Integer[] si = new Integer[F];
            for (int i = 0; i < F; i++) si[i] = i;
            Arrays.sort(si, (a, b) -> Double.compare(korArr[b][1], korArr[a][1]));

            System.out.println("Top 5 özellik:");
            for (int i = 0; i < 5; i++)
                System.out.printf("  %d. %-15s: r=%.4f\n", i + 1, FN[si[i]], korArr[si[i]][0]);

            System.out.println("\n=== VERİ KEŞFİ VE ÖN İŞLEME TAMAMLANDI ===");

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
