package com.btkakademi.ml.bolum08;

import org.apache.commons.csv.CSVFormat;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.model.cart.SplitRule;
import smile.validation.ClassificationMetrics;

import java.nio.file.Paths;

public class SmileKararAgaci {
    public static void main(String[] args) {
        try {
            var url = SmileKararAgaci.class.getClassLoader().getResource("datasets/iris.csv");

            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .build();

            DataFrame veri = Read.csv(Paths.get(url.toURI()), format);

            veri = veri.factorize("class");

            // Formula Tanımlama
            String hedefSutun = veri.names()[veri.ncol() - 1];
            Formula formula = Formula.lhs(hedefSutun);

            System.out.println("Hedef: " + hedefSutun);

            // Random Forest Model Eğitimi
            long baslangic = System.currentTimeMillis();

            //RandomForest Options
            var options = new RandomForest.Options(
                    100, // ağaç sayısı
                    0, // mtry=0, sqrt(p) otomatik
                    SplitRule.GINI, // bölme kriteri
                    20, // maxDepth
                    0, // maxNodes=0 sınırsız
                    5, // nodeSize = yapraktaki min örnek
                    1.0, // subsample: boostrap oranı
                    null, //classWeight
                    null, //seeds
                    null //controller
            );

            // Modeli Eğit
            var model = RandomForest.fit(formula, veri, options);

            long sure = System.currentTimeMillis() - baslangic;
            System.out.println("Model egitildi (" + sure + " ms");
            System.out.println("Agac Sayisi: " + model.size());

            // OOB Metrikleri
            System.out.println("OOB Metrikleri");
            ClassificationMetrics metrics = model.metrics();
            if (metrics != null) {
                System.out.println("OOB Doğruluk: " + metrics.accuracy() * 100);
                System.out.println("OOB Hata: " + (1 - metrics.accuracy()) * 100);
            }

            // Özellik Önemliliği
            System.out.println("\nOzellik Onemliligi ");
            double[] importance = model.importance();
            String[] ozellikler = {"sepalLength", "sepalWidth", "petalLength", "petalWidth"};

            Integer[] sirali = new Integer[ozellikler.length];
            for (int i = 0; i < sirali.length; i++) sirali[i] = i;
            java.util.Arrays.sort(sirali, (a, b) -> Double.compare(importance[b], importance[a]));

            for (int i = 0; i < sirali.length; i++) {
                int idx = sirali[i];
                int barLen = (int) (importance[idx] * 2);
                String bar = "*".repeat(Math.max(1, barLen));
                System.out.printf("%d. %-14s: %6.2f %s%n", i + 1, ozellikler[idx], importance[idx], bar);
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
