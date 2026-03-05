package com.btkakademi.ml.bolum07;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LinearModel;
import smile.regression.OLS;

import java.nio.file.Path;

public class MetrikYorumlama {
    static void main() {
        System.out.println("AUTO-MPG için Yakıt Tüketimi Tahmini");
        hesaplaVeYorumla("src/main/resources/datasets/auto-mpg.csv", "mpg");

        System.out.println("WINE QUALITY için Şarap Kalitesi Tahmini");
        hesaplaVeYorumla("src/main/resources/datasets/winequality-red.csv", "quality");
    }

    static void hesaplaVeYorumla(String dosyaYolu, String hedefSutun) {
        try {
            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader().setSkipHeaderRecord(true).setIgnoreHeaderCase(true).setTrim(true).build();

            DataFrame veriSeti = Read.csv(Path.of(dosyaYolu), format);
            int n = veriSeti.nrow();

            int hedefIndex = 0;
            for (int i = 0; i < veriSeti.ncol(); i++) {
                if (veriSeti.column(i).name().equalsIgnoreCase(hedefSutun)) {
                    hedefIndex = i;
                    break;
                }
            }

            Formula formula = Formula.lhs(hedefSutun);
            LinearModel model = OLS.fit(formula, veriSeti);

            // Metrikleri Hesaplama
            double toplamMutlak = 0, toplamKare = 0, toplamGercek = 0, hedefMin = Double.MAX_VALUE, hedefMax = Double.MIN_VALUE;

            for (int i = 0; i < n; i++) {
                double gercek = veriSeti.getDouble(i, hedefIndex);
                double tahmin = model.predict(veriSeti.get(i));
                double hata = gercek - tahmin;

                toplamMutlak += Math.abs(hata);
                toplamKare += hata * hata;
                toplamGercek += gercek;

                if (gercek < hedefMin) hedefMin = gercek;
                if (gercek > hedefMax) hedefMax = gercek;
            }

            double mae = toplamGercek / n;
            double rmse = Math.sqrt(toplamKare / n);
            double gercekOrt = toplamGercek / n;

            double ssTot = 0;
            for (int i = 0; i < n; i++) {
                double gercek = veriSeti.getDouble(i, hedefIndex);
                ssTot += Math.pow(gercek - gercekOrt, 2);
            }

            double r2 = 1 - (toplamKare / ssTot);

            System.out.println("Örnek " + n + "Hedef: " + hedefSutun);
            System.out.println("Hedef araligi= Min:" + hedefMin + ", Max: " + hedefMax);

            System.out.println("R2 değeri: " + r2);
            if (r2 >= 0.7) System.out.println("İyi bir sonuç, varyansın %" + r2 * 100 + "'sını açıklıyor.");
            else if (r2 >= 0.5) System.out.println("R2: Orta");
            else System.out.println("Model yetersiz");

            // RMSE-MAE Karşılaştırması
            double fark = (rmse - mae) / mae * 100;
            System.out.println("RMSE: " + rmse);
            if (fark < 20) System.out.println("Hatalar homojendir.");
            else System.out.println("Büyük hatalar var. (outlier etkisi.)");

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
