package com.btkakademi.ml.bolum10;

import org.apache.commons.csv.CSVFormat;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.model.cart.SplitRule;
import smile.validation.ClassificationMetrics;

import java.nio.file.Paths;
import java.util.Arrays;

public class SmileRandomForest {
    static void main() {
        System.setProperty("org.slf4j.simpleLogger.log.smile", "warn");
        try {
            var url = SmileRandomForest.class.getClassLoader().getResource("datasets/iris.csv");

            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .get();

            DataFrame veri = Read.csv(Paths.get(url.toURI()), format);

            veri = veri.factorize("class");

            // Forumla
            Formula formula = Formula.lhs("class");

            // RandomForest Parametreleri
            int ntress = 100; // Kaç tane karar ağacı eğitilecek
            int mtry = 0; // Her düğüm bölünümde kaç ozellik denecek
            int maxDepth = 20; // Ağacların maksimum derinliği
            int maxNodes = 0; // Maksimum yaprak sayisi
            int nodeSize = 5; // Yapraktaki minimum ornek sayisi
            double subsample = 1.0; // Bootstrap ornekleme orani

            // Model Eğitimi
            var options = new RandomForest.Options(
                    ntress,
                    mtry,
                    SplitRule.GINI,
                    maxDepth,
                    maxNodes,
                    nodeSize,
                    subsample,
                    null, null, null
            );

            long baslangic = System.currentTimeMillis();
            var model = RandomForest.fit(formula, veri, options);
            long sure = System.currentTimeMillis() - baslangic;

            System.out.println("Modelimiz egitildi, sure: " + sure + "ms");
            System.out.println("Ağac Sayısı: " + model.size());

            // OOB HATA METRİKLERİ
            ClassificationMetrics metrics = model.metrics();
            if (metrics != null) {
                double oobDogruluk = metrics.accuracy();
                System.out.println("Dogruluk Oranı: " + oobDogruluk * 100);
            }

            // Özellik Önemliliği
            String[] ozellikler = {"sepalLength", "sepalWidth", "petalLength", "petalWidth"};
            double[] importance = model.importance();

            Integer[] sirali = new Integer[ozellikler.length];
            for (int i = 0; i < sirali.length; i++) sirali[i] = i;

            Arrays.sort(sirali, (a, b) -> Double.compare(importance[b], importance[a]));
            System.out.println("Ozellik Onemliligi");
            for (int i = 0; i < sirali.length; i++) {
                int sira = sirali[i];
                System.out.printf(" %d. %-14s: %.4f\n", i + 1, ozellikler[sira], importance[sira]);
            }


            // Ağaç Sayısı Etkisi
            System.out.println("Agac Sayisi Etkisi");
            int[] agacSayilari = {10, 50, 100, 200, 300};
            for (int n : agacSayilari) {
                var testOptions = new RandomForest.Options(
                        n, 0, SplitRule.GINI, 20, 0, 5, 1.0, null, null, null);

                var testModel = RandomForest.fit(formula, veri, testOptions);
                ClassificationMetrics testMetrics = testModel.metrics();
                double acc = testMetrics != null ? testMetrics.accuracy() : 0;
                System.out.printf("n=%3d: OOB=%.2f%%\n", n, acc * 100);
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
