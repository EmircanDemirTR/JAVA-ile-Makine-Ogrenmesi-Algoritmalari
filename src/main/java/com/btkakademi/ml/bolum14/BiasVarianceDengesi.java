package com.btkakademi.ml.bolum14;

import smile.classification.KNN;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class BiasVarianceDengesi {
    static void main() {
        try {
            var is = BiasVarianceDengesi.class.getClassLoader().getResourceAsStream("datasets/wine.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                etiketler.add(Integer.parseInt(p[0].trim()) - 1);

                ozellikler.add(new double[]{
                        Double.parseDouble(p[1].trim()), // alkol
                        Double.parseDouble(p[10].trim()) // renk yoğunluğu
                });
            }

            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();
            int n = X.length;

            X = normalize(X);

            // Bootstrap Deneyi Tasarımı
            int bootstrapSayisi = 50;
            int egitimBoyut = (int) (0.7 * n);
            Random rng = new Random(42);

            int testBoyut = 30;
            Random testRng = new Random(99);
            Set<Integer> testSet = new LinkedHashSet<>();
            while (testSet.size() < testBoyut) {
                testSet.add(testRng.nextInt(n));
            }
            int[] sabitTestIdx = testSet.stream().mapToInt(i -> i).sorted().toArray();
            System.out.println("Bootstrap Sayısı: " + bootstrapSayisi);
            System.out.println("Sabit Test Noktası : " + testBoyut);

            // YÜKSEK BİAS MODELİ KNN K=60
            int[][] tahminlerBasit = new int[bootstrapSayisi][testBoyut];

            for (int b = 0; b < bootstrapSayisi; b++) {
                // Rastgele eğitim seti
                int[] trainIdx = rastgeleEgitimSec(n, egitimBoyut, sabitTestIdx, rng);
                double[][] XTrain = new double[trainIdx.length][];
                int[] yTrain = new int[trainIdx.length];
                for (int i = 0; i < trainIdx.length; i++) {
                    XTrain[i] = X[trainIdx[i]];
                    yTrain[i] = y[trainIdx[i]];
                }

                var model = KNN.fit(XTrain, yTrain, 60);
                for (int t = 0; t < testBoyut; t++) {
                    tahminlerBasit[b][t] = model.predict(X[sabitTestIdx[t]]);
                }
            }

            double[] bvBasit = biasVarianceHesapla(tahminlerBasit, sabitTestIdx, y);
            System.out.printf("  Ortalama Bias2:      %.4f%n", bvBasit[0]);
            System.out.printf("  Ortalama Vary.:      %.4f%n", bvBasit[1]);
            System.out.printf("  Ortalama Doğr.:      %.4f%n", bvBasit[2] * 100);


            // YÜKSEK VARYANS MODELİ KNN K=1
            int[][] tahminlerKarisik = new int[bootstrapSayisi][testBoyut];

            for (int b = 0; b < bootstrapSayisi; b++) {
                // Rastgele eğitim seti
                int[] trainIdx = rastgeleEgitimSec(n, egitimBoyut, sabitTestIdx, rng);
                double[][] XTrain = new double[trainIdx.length][];
                int[] yTrain = new int[trainIdx.length];
                for (int i = 0; i < trainIdx.length; i++) {
                    XTrain[i] = X[trainIdx[i]];
                    yTrain[i] = y[trainIdx[i]];
                }

                var model = KNN.fit(XTrain, yTrain, 1);
                for (int t = 0; t < testBoyut; t++) {
                    tahminlerKarisik[b][t] = model.predict(X[sabitTestIdx[t]]);
                }
            }

            double[] bvKarisik = biasVarianceHesapla(tahminlerKarisik, sabitTestIdx, y);
            System.out.printf("  Ortalama Bias2:      %.4f%n", bvKarisik[0]);
            System.out.printf("  Ortalama Vary.:      %.4f%n", bvKarisik[1]);
            System.out.printf("  Ortalama Doğr.:      %.4f%n", bvKarisik[2] * 100);


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double[][] normalize(double[][] X) {
        int n = X.length;
        int m = X[0].length;
        double[] min = new double[m];
        double[] max = new double[m];

        for (int j = 0; j < m; j++) {
            min[j] = Double.MAX_VALUE;
            max[j] = Double.MIN_VALUE;
            for (int i = 0; i < n; i++) {
                if (X[i][j] < min[j]) min[j] = X[i][j];
                if (X[i][j] > max[j]) max[j] = X[i][j];
            }
        }

        double[][] X_norm = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                X_norm[i][j] = (X[i][j] - min[j]) / (max[j] - min[j] + 1e-10); // avoid division by zero
            }
        }

        return X_norm;
    }

    private static int[] rastgeleEgitimSec(int n, int egitimBoyut, int[] haricIdx, Random rng) {
        Set<Integer> haricSet = new HashSet<>();
        for (int idx : haricIdx) {
            haricSet.add(idx);
        }

        List<Integer> kullanilabilir = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (!haricSet.contains(i)) {
                kullanilabilir.add(i);
            }
        }

        Collections.shuffle(kullanilabilir, rng);
        int boyut = Math.min(egitimBoyut, kullanilabilir.size());
        int[] sonuc = new int[boyut];
        for (int i = 0; i < boyut; i++) {
            sonuc[i] = kullanilabilir.get(i);
        }
        return sonuc;
    }

    private static double[] biasVarianceHesapla(int[][] tahminler, int[] testIdx, int[] gercekEtiket) {
        int bootstrapSayisi = tahminler.length;
        int testBoyut = tahminler[0].length;

        double toplamBias = 0.0;
        double toplamVariance = 0.0;
        double toplamDogruSayisi = 0;

        for (int t = 0; t < testBoyut; t++) {
            int gercek = gercekEtiket[testIdx[t]];

            Map<Integer, Integer> tahminSayilari = new HashMap<>();
            for (int b = 0; b < bootstrapSayisi; b++) {
                tahminSayilari.merge(tahminler[b][t], 1, Integer::sum);
            }

            int mod = -1;
            int maxSayi = 0;
            for (var entry : tahminSayilari.entrySet()) {
                if (entry.getValue() > maxSayi) {
                    maxSayi = entry.getValue();
                    mod = entry.getKey();
                }
            }

            toplamBias += (mod != gercek) ? 1.0 : 0.0;
            int modFarkSayisi = 0;
            for (int b = 0; b < bootstrapSayisi; b++) {
                if (tahminler[b][t] != mod) {
                    modFarkSayisi++;
                }
            }
            toplamVariance += (double) modFarkSayisi / bootstrapSayisi;

            int dogruSayisi = 0;
            for (int b = 0; b < bootstrapSayisi; b++) {
                if (tahminler[b][t] == gercek) {
                    dogruSayisi++;
                }
            }
            toplamDogruSayisi += (double) dogruSayisi / bootstrapSayisi;
        }
        return new double[]{
                toplamBias / testBoyut,
                toplamVariance / testBoyut,
                toplamDogruSayisi / testBoyut
        };
    }
}
