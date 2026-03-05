package com.btkakademi.ml.bolum08;

import smile.classification.LogisticRegression;
import smile.validation.metric.Accuracy;
import smile.validation.metric.ConfusionMatrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmileLojistikRegresyon {
    static void main() {
        try {
            var is = SmileLojistikRegresyon.class.getClassLoader().getResourceAsStream("datasets/glass.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            String header = reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            Map<Integer, Integer> sinifMap = new LinkedHashMap<>();
            sinifMap.put(1, 0);
            sinifMap.put(2, 1);
            sinifMap.put(3, 2);
            sinifMap.put(5, 3);
            sinifMap.put(6, 4);
            sinifMap.put(7, 5);

            String[] sinifAdlari = {
                    "Bina Cami (Float)",
                    "Bina Cami (Non-Float)",
                    "Arac Cami",
                    "Konteyner",
                    "Sofra Takimi",
                    "Far Cami"
            };

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");

                double[] oz = new double[9];
                for (int i = 0; i < 9; i++) {
                    oz[i] = Double.parseDouble(p[i].trim());
                }
                ozellikler.add(oz);

                int sinif = Integer.parseInt(p[9].trim());
                etiketler.add(sinifMap.get(sinif));
            }

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();

            // Sınıf dağılımı hesabı
            System.out.println("Sinif Dagilimi: ");
            int[] sinifSayilari = new int[6];
            for (int yi : y) sinifSayilari[yi]++;
            for (int i = 0; i < sinifAdlari.length; i++) {
                System.out.printf(" %d. %-18s: %d ornek%n", i, sinifAdlari[i], sinifSayilari[i]);
            }

            // Train / Test Split
            int n = X.length;
            int egitimBoyut = (int) (n * 0.7);
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;

            Random rand = new Random(42);

            for (int i = n - 1; i > 0; i--) {
                int j = rand.nextInt(i + 1);
                int temp = idx[i];
                idx[i] = idx[j];
                idx[j] = temp;
            }

            double[][] XTrain = new double[egitimBoyut][];
            double[][] XTest = new double[n - egitimBoyut][];
            int[] yTrain = new int[egitimBoyut];
            int[] yTest = new int[n - egitimBoyut];

            for (int i = 0; i < egitimBoyut; i++) {
                XTrain[i] = X[idx[i]];
                yTrain[i] = y[idx[i]];
            }

            for (int i = egitimBoyut; i < n; i++) {
                XTest[i - egitimBoyut] = X[idx[i]];
                yTest[i - egitimBoyut] = y[idx[i]];
            }

            System.out.println("Egitim Seti: " + XTrain.length);
            System.out.println("Test Seti: " + XTest.length);

            // Model Egitimi:
            long baslangic = System.currentTimeMillis();
            var model = LogisticRegression.fit(XTrain, yTrain);
            long sure = System.currentTimeMillis() - baslangic;
            System.out.println("Model egitildi ve sure: " + sure + "ms");

            // Tahmin ve Değerlendirme
            int[] tahminler = model.predict(XTest);
            double dogruluk = Accuracy.of(yTest, tahminler);

            System.out.println("Dogruluk Orani: " + dogruluk * 100);

            // Karmaşıklık Matrisi
            var cm = ConfusionMatrix.of(yTest, tahminler);
            System.out.println("Satirlar: Gercek, Sutunlar: Tahmin");
            System.out.println(cm);

            // Sınıf Bazında Analiz

            int numClasses = sinifAdlari.length;
            int[] sinifToplam = new int[numClasses];
            int[] sinifDogru = new int[numClasses];

            for (int i = 0; i < yTest.length; i++) {
                int gercekSinif = yTest[i];
                int tahminSinif = tahminler[i];

                if (gercekSinif < numClasses) {
                    sinifToplam[gercekSinif]++;
                    if (gercekSinif == tahminSinif) {
                        sinifDogru[gercekSinif]++;
                    }
                }
            }

            for (int i = 0; i < numClasses; i++) {
                if (sinifToplam[i] > 0) {
                    double basari = (sinifDogru[i] * 100.0 / sinifToplam[i]);
                    System.out.printf("  %-18s: %d/%d dogru (%.1f%%)%n",
                            sinifAdlari[i], sinifDogru[i], sinifToplam[i], basari);
                } else {
                    System.out.printf("  %-18s: Test setinde yok%n", sinifAdlari[i]);
                }
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
