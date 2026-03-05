package com.btkakademi.ml.bolum14;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LASSO;
import smile.regression.LinearModel;
import smile.regression.OLS;
import smile.regression.RidgeRegression;
import smile.util.Index;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class SmileRidgeLasso {

    // METODU SINIFIN İÇİNE TAŞIDIK
    private static double[] hesaplaMetrikler(LinearModel model, DataFrame veri, String hedef) {
        int n = veri.nrow();
        int hedefIdx = veri.schema().indexOf(hedef);
        double toplamKareHata = 0;
        double toplamKareSapma = 0;
        double toplamMutlakHata = 0;
        double toplam = 0;

        // Ortalamayi hesapla
        for (int i = 0; i < n; i++) {
            toplam += veri.getDouble(i, hedefIdx);
        }
        double ortalama = toplam / n;

        // Metrikleri hesapla
        for (int i = 0; i < n; i++) {
            double gercek = veri.getDouble(i, hedefIdx);
            double tahmin = model.predict(veri.get(i));
            double hata = gercek - tahmin;
            toplamKareHata += hata * hata;
            toplamKareSapma += (gercek - ortalama) * (gercek - ortalama);
            toplamMutlakHata += Math.abs(hata);
        }

        double rmse = Math.sqrt(toplamKareHata / n);
        double r2 = 1 - (toplamKareHata / toplamKareSapma);
        double mae = toplamMutlakHata / n;
        return new double[]{rmse, r2, mae};
    }

    // Main metodu standart imzasına (String[] args) kavuşturduk
    public static void main(String[] args) {
        try {
            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .get();

            // Dosya yolunu kontrol etmeyi unutma
            var url = SmileRidgeLasso.class.getClassLoader().getResource("datasets/winequality-red.csv");
            if (url == null) {
                System.err.println("Dosya bulunamadı! 'resources/datasets/winequality-red.csv' yolunu kontrol et.");
                return;
            }

            DataFrame veri = Read.csv(Path.of(url.toURI()), format);

            // Train Test Split
            int n = veri.nrow();
            int egitimBoyut = (int) (n * 0.7);
            int testBoyut = n - egitimBoyut;

            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) {
                idx[i] = i;
            }
            Collections.shuffle(Arrays.asList(idx), new Random(42));

            int[] trainIdx = new int[egitimBoyut];
            int[] testIdx = new int[testBoyut];
            for (int i = 0; i < egitimBoyut; i++) trainIdx[i] = idx[i];
            for (int i = 0; i < testBoyut; i++) testIdx[i] = idx[i + egitimBoyut];

            DataFrame trainVeri = veri.get(Index.of(trainIdx));
            DataFrame testVeri = veri.get(Index.of(testIdx));

            Formula formula = Formula.lhs("quality");

            // --- OLS Baseline ---
            LinearModel olsModel = OLS.fit(formula, trainVeri);
            double[] olsTrainMetrik = hesaplaMetrikler(olsModel, trainVeri, "quality");
            double[] olsTestMetrik = hesaplaMetrikler(olsModel, testVeri, "quality");

            System.out.println("--- OLS Model ---");
            System.out.printf(" Train RMSE: %.4f  |  Test RMSE: %.4f%n", olsTrainMetrik[0], olsTestMetrik[0]);
            System.out.printf(" Train R2:   %.4f  |  Test R2:   %.4f%n", olsTrainMetrik[1], olsTestMetrik[1]);
            System.out.printf(" Train MAE:  %.4f  |  Test MAE:  %.4f%n", olsTrainMetrik[2], olsTestMetrik[2]);

            // --- RIDGE Regression ---
            double ridgeLambda = 100.0;
            LinearModel ridgeModel = RidgeRegression.fit(formula, trainVeri, ridgeLambda);
            double[] ridgeTrainMetrik = hesaplaMetrikler(ridgeModel, trainVeri, "quality");
            double[] ridgeTestMetrik = hesaplaMetrikler(ridgeModel, testVeri, "quality");

            System.out.printf("%nRIDGE Regression (L2) - Lambda: %.2f%n", ridgeLambda);
            System.out.printf(" Train RMSE: %.4f  |  Test RMSE: %.4f%n", ridgeTrainMetrik[0], ridgeTestMetrik[0]);

            // --- LASSO Regression ---
            System.out.println("\n--- LASSO Path ---");
            System.out.printf("%10s %10s %10s %10s %s%n", "Lambda", "RMSE", "R2", "MAE", "Sifir/Top Katsayi");

            double[] lassoLambdalar = {0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0};

            double enIyiLassoRmse = Double.MAX_VALUE;
            double enIyiLassoLambda = 0;
            int enIyiLassoSifir = 0;

            for (double lambda : lassoLambdalar) {
                var options = new LASSO.Options(lambda);
                LinearModel model = LASSO.fit(formula, trainVeri, options);
                double[] testMetrik = hesaplaMetrikler(model, testVeri, "quality");

                var katsayilar = model.coefficients();
                int sifir = 0;
                for (int i = 0; i < katsayilar.size(); i++) {
                    if (Math.abs(katsayilar.get(i)) < 1e-6) sifir++;
                }

                if (testMetrik[0] < enIyiLassoRmse) {
                    enIyiLassoRmse = testMetrik[0];
                    enIyiLassoLambda = lambda;
                    enIyiLassoSifir = sifir;
                }

                System.out.printf("  %-10.2f  %10.4f  %10.4f  %10.4f  %d / %d%n",
                        lambda, testMetrik[0], testMetrik[1], testMetrik[2],
                        sifir, katsayilar.size());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}