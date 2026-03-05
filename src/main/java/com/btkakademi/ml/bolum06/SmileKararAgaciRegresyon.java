package com.btkakademi.ml.bolum06;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.RegressionTree;

import java.nio.file.Path;

public class SmileKararAgaciRegresyon {

    static void main() {
        try {
            String dosyaYolu = "src/main/resources/datasets/auto-mpg.csv";

            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .get();

            DataFrame veriSeti = Read.csv(Path.of(dosyaYolu), format);
            System.out.println("Veri Seti: " + veriSeti.nrow());

            // Formül tanımlama
            Formula formula = Formula.lhs("mpg");

            // RegressionTree
            //maxDepth 20 - maxNodes 0, nodeSize 5
            RegressionTree model = RegressionTree.fit(formula, veriSeti);


            // Ağaç Yapısını Görme
            String agacStr = model.toString();
            if (agacStr.length() > 2000) {
                System.out.println(agacStr.substring(0, 2000) + "\n... (Devamı kısaltıldı)");
            } else {
                System.out.println(agacStr);
            }

            // Tahmin Örnekleri
            System.out.println("Tahmin Örnekleri: ");

            int mpgIndex = 5;
            for (int i = 0; i < Math.min(10, veriSeti.nrow()); i++) {
                double gercek = veriSeti.getDouble(i, mpgIndex);
                double tahmin = model.predict(veriSeti.get(i));
                double hata = gercek - tahmin;
                System.out.printf("%6.1f | %6.1f | %+6.1f%n", gercek, tahmin, hata);
            }


            // Model Performansı
            double toplamKareHata = 0;
            double toplamGercek = 0;
            double toplamGercekKare = 0;
            int n = veriSeti.nrow();

            for (int i = 0; i < n; i++) {
                double gercek = veriSeti.getDouble(i, mpgIndex);
                double tahmin = model.predict(veriSeti.get(i));
                double hata = gercek - tahmin;

                toplamKareHata += hata * hata;
                toplamGercek += gercek;
                toplamGercekKare += gercek * gercek;
            }

            double rmse = Math.sqrt(toplamKareHata / n);

            double ortalama = toplamGercek / n;
            double ssTot = toplamGercekKare - (toplamGercek * toplamGercek / n);
            double ssRes = toplamKareHata;
            double r2 = 1 - (ssRes / ssTot);

            System.out.println("R2: " + r2);
            System.out.println("RMSE: " + rmse);

            // Özellik Önemliliği
            System.out.println("\n\nÖzellik Önemliliği: ");
            double[] importance = model.importance();
            String[] ozellikler = {"cylinders", "displacement", "horsepower", "weight", "acceleration"};

            for (int i = 0; i < Math.min(ozellikler.length, importance.length); i++) {
                System.out.printf("%s: %.4f%n", ozellikler[i], importance[i]);
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }
}
