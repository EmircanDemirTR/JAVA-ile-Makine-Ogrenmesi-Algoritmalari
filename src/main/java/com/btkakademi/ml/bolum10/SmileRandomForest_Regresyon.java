package com.btkakademi.ml.bolum10;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.RandomForest;

import java.nio.file.Paths;

public class SmileRandomForest_Regresyon {
    static void main() {
        try {
            var url = SmileRandomForest_Regresyon.class.getClassLoader().getResource("datasets/auto-mpg.csv");

            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .get();

            DataFrame veri = Read.csv(Paths.get(url.toURI()), format);

            String[] sutunlar = veri.names();

            int mpgIdx = -1;
            for (int i = 0; i < sutunlar.length; i++) {
                if (sutunlar[i].equalsIgnoreCase("mpg")) {
                    mpgIdx = i;
                    break;
                }
            }

            double minMpg = Double.MAX_VALUE;
            double maxMpg = Double.MIN_VALUE;
            double sumMpg = 0;
            for (int i = 0; i < veri.nrow(); i++) {
                double mpg = veri.getDouble(i, mpgIdx);
                if (mpg < minMpg) minMpg = mpg;
                if (mpg > maxMpg) maxMpg = mpg;
                sumMpg += mpg;
            }

            double avgMpg = sumMpg / veri.nrow();
            System.out.println("MPG Min: " + minMpg);
            System.out.println("MPG Max: " + maxMpg);
            System.out.println("MPG Ort: " + avgMpg);


            // Model Paremetreleri
            Formula formula = Formula.lhs("mpg");

            int ntress = 100;
            int mtry = 0;
            int maxDepth = 20;
            int maxNodes = 0;
            int nodeSize = 5;
            double subsample = 1.0;

            long baslangic = System.currentTimeMillis();
            var options = new RandomForest.Options(
                    ntress,
                    mtry,
                    maxDepth,
                    maxNodes,
                    nodeSize,
                    subsample,
                    null,
                    null
            );
            var model = RandomForest.fit(formula, veri, options);
            long sure = System.currentTimeMillis() - baslangic;

            System.out.println("Model egitildi! Sure: " + sure + "ms");


            // Tahminler ve Metrikler
            double[] tahminler = model.predict(veri);

            double[] gercekler = new double[veri.nrow()];
            for (int i = 0; i < veri.nrow(); i++) {
                gercekler[i] = veri.getDouble(i, mpgIdx);
            }

            double maeSum = 0;
            for (int i = 0; i < gercekler.length; i++) {
                maeSum += Math.abs(gercekler[i] - tahminler[i]);
            }
            double mae = maeSum / gercekler.length;

            double mseSum = 0;
            for (int i = 0; i < gercekler.length; i++) {
                double hata = gercekler[i] - tahminler[i];
                mseSum += hata * hata;
            }
            double rmse = Math.sqrt(mseSum / gercekler.length);

            double ssRes = 0, ssTot = 0;
            for (int i = 0; i < gercekler.length; i++) {
                ssRes += Math.pow(gercekler[i] - tahminler[i], 2);
                ssTot += Math.pow(gercekler[i] - avgMpg, 2);
            }
            double r2 = 1 - (ssRes / ssTot);

            System.out.println("MAE: " + mae);
            System.out.println("RMSE: " + rmse);
            System.out.println("R2: " + r2);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
