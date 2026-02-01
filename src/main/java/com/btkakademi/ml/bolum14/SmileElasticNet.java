package com.btkakademi.ml.bolum14;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.*;
import smile.util.Index;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class SmileElasticNet {
    static void main() {
        try {
            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .get();

            // Dosya yolunu kontrol etmeyi unutma
            var url = SmileElasticNet.class.getClassLoader().getResource("datasets/winequality-red.csv");
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

            Formula formula = formula = Formula.lhs("quality");

            // Elastic Net Modeli
            double lambda1 = 5.0;
            double lambda2 = 5.0;
            LinearModel enModel = ElasticNet.fit(formula, trainVeri, lambda1, lambda2);

            double[] enTrainM = hesaplaMetrikler(enModel, trainVeri, "quality");
            double[] enTestM = hesaplaMetrikler(enModel, testVeri, "quality");

            System.out.printf(" Elastic Net%n");
            System.out.printf(" Train RMSE: %.4f  |  Test RMSE: %.4f%n", enTrainM[0], enTestM[0]);
            System.out.printf(" Train R2:   %.4f  |  Test R2:   %.4f%n", enTrainM[1], enTestM[1]);
            System.out.printf(" Train MAE:  %.4f  |  Test MAE:  %.4f%n", enTrainM[2], enTestM[2]);

            var enKatsayilar = enModel.coefficients();
            String[] ozellikler = veri.names();
            int sifirSayisi = 0;
            for (int i = 0; i < enKatsayilar.size(); i++) {
                if (Math.abs(enKatsayilar.get(i)) < 1e-6) {
                    sifirSayisi++;
                }
            }
            System.out.println("Sıfır olan katsayı sayısı: " + sifirSayisi);

            // Model Karşılaştırma
            // 1. OLS - BASELINE
            LinearModel olsModel = OLS.fit(formula, trainVeri);
            double[] olsTestM = hesaplaMetrikler(olsModel, testVeri, "quality");

            // 2. RIDGE LAMBDA=10
            LinearModel ridgeModel = RidgeRegression.fit(formula, trainVeri, 10.0);
            double[] ridgeTestM = hesaplaMetrikler(ridgeModel, testVeri, "quality");

            // 3. LASSO LAMBDA=10
            LinearModel lassoModel = LASSO.fit(formula, trainVeri, new LASSO.Options(10.0));
            double[] lassoTestM = hesaplaMetrikler(lassoModel, testVeri, "quality");

            // 4. ELASTIC NET LAMBDA1=5, LAMBDA2=5
            LinearModel elasticModel = ElasticNet.fit(formula, trainVeri, 5.0, 5.0);
            double[] elasticTestM = hesaplaMetrikler(elasticModel, testVeri, "quality");

            System.out.println("\nModel Karsilastirma (Test Seti)");
            System.out.printf("%-15s %-10s %-10s %-10s%n", "Model", "RMSE", "R2", "MAE");
            System.out.printf("%-15s %-10.4f %-10.4f %-10.4f%n", "OLS", olsTestM[0], olsTestM[1], olsTestM[2]);
            System.out.printf("%-15s %-10.4f %-10.4f %-10.4f%n", "Ridge (L2=10)", ridgeTestM[0], ridgeTestM[1], ridgeTestM[2]);
            System.out.printf("%-15s %-10.4f %-10.4f %-10.4f%n", "Lasso (L1=10)", lassoTestM[0], lassoTestM[1], lassoTestM[2]);
            System.out.printf("%-15s %-10.4f %-10.4f %-10.4f%n", "Elastic Net", elasticTestM[0], elasticTestM[1], elasticTestM[2]);


            // KATSAYI KARŞILAŞTIRMA
            var olsK = olsModel.coefficients();
            var ridgeK = ridgeModel.coefficients();
            var lassoK = lassoModel.coefficients();
            var elasticK = elasticModel.coefficients();

            System.out.println("\nKatsayi Karsilastirma");
            System.out.printf("%-15s %-10s %-10s %-10s %-10s%n", "Ozellik", "OLS", "Ridge", "Lasso", "Elastic");
            for (int i = 0; i < olsK.size(); i++) {
                System.out.printf("%-15s %-10.4f %-10.4f %-10.4f %-10.4f%n",
                        ozellikler[i],
                        olsK.get(i),
                        ridgeK.get(i),
                        lassoK.get(i),
                        elasticK.get(i));
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

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

}
