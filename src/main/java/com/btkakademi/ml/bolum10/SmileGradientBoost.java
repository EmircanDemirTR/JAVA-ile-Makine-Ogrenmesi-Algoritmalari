package com.btkakademi.ml.bolum10;

import org.apache.commons.csv.CSVFormat;
import smile.classification.GradientTreeBoost;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.validation.metric.Accuracy;

import java.nio.file.Paths;

public class SmileGradientBoost {
    static void main() {
        try {
            var url = SmileGradientBoost.class.getClassLoader().getResource("datasets/wine.csv");

            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .get();

            DataFrame veri = Read.csv(Paths.get(url.toURI()), format);
            veri = veri.factorize("class");

            String[] sutunlar = veri.names();
            Formula formula = Formula.lhs("class");

            // Gradient Boosting Parametreleri
            int ntrees = 100; // Boosting adim sayisi
            int maxDepth = 6; //ağaç derinliği
            int maxNodes = 32; // Maksimum yaprak sayisi
            int nodeSize = 5; // yapraktaki minimum ornek sayisi
            double shrinkage = 0.1; //Learning rate / ogrenme orani
            // Genel kural: Shrinkage dusuk, ntrees yuksek
            double subsample = 0.8; // Her ağaç için kullanılan veri oranı

            // Model Eğitimi
            long baslangic = System.currentTimeMillis();
            var options = new GradientTreeBoost.Options(
                    ntrees,
                    maxDepth,
                    maxNodes,
                    nodeSize,
                    shrinkage,
                    subsample,
                    null,
                    null
            );

            var model = GradientTreeBoost.fit(formula, veri, options);
            long sure = System.currentTimeMillis() - baslangic;
            System.out.println("Model Egitildi: " + sure + "ms");


            // Eğitim Doğruluğu
            int[] gercekEtiketler = formula.y(veri).toIntArray();

            int[] tahminler = model.predict(veri);

            double dogruluk = Accuracy.of(gercekEtiketler, tahminler);
            System.out.println("Egitim dogrulugu: " + dogruluk * 100);

            // Learning Rate Etkisi
            double[] lrDegerleri = {0.01, 0.05, 0.1, 0.2, 0.3};

            for (double lr : lrDegerleri) {
                var lrOptions = new GradientTreeBoost.Options(
                        ntrees,
                        maxDepth,
                        maxNodes,
                        nodeSize,
                        lr,
                        subsample,
                        null,
                        null
                );

                var lrModel = GradientTreeBoost.fit(formula, veri, lrOptions);
                int[] lrTahminler = lrModel.predict(veri);
                double lrAcc = Accuracy.of(gercekEtiketler, lrTahminler);
                System.out.println("LR Degeri: " + lr + " icin Dogruluk Degeri: " + lrAcc * 100);

            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
