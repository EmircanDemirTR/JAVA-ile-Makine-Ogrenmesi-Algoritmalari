package com.btkakademi.ml.bolum06;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LinearModel;
import smile.regression.OLS;

import java.nio.file.Path;

public class SmilePolinomRegresyon {
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

            // Veriyi Hazırlayalım

            int n = veriSeti.nrow(); // Toplam Örnek Sayısı
            int hpIndex = 2; // Horsepower sütun indeksi
            int mpgIndex = 5; // MPG'nin sütun indeksi


            // Değerleri Dizilere Çıkar
            double[] hp = new double[n];
            double[] mpg = new double[n];

            for (int i = 0; i < n; i++) {
                hp[i] = veriSeti.getDouble(i, hpIndex);
                mpg[i] = veriSeti.getDouble(i, mpgIndex);
            }

            // 1. DERECE LINEER MODEL

            double[][] lineerData = new double[n][2];
            for (int i = 0; i < n; i++) {
                lineerData[i][0] = hp[i]; // X
                lineerData[i][1] = mpg[i]; // Y
            }
            DataFrame lineerVeri = DataFrame.of(lineerData, "hp", "mpg");

            Formula formula = Formula.lhs("mpg");
            LinearModel lineerModel = OLS.fit(formula, lineerVeri);

            double linB0 = lineerModel.intercept();
            double linB1 = lineerModel.coefficients().get(0);

            System.out.println("\n\n----MODEL 1: LİNEER MODEL----");
            System.out.printf("mpg= %.4f + (%.4f) * hp%n", linB0, linB1);


            // 2. DERECE QUADRATIC MODEL
            double[][] quadData = new double[n][3];
            for (int i = 0; i < n; i++) {
                quadData[i][0] = hp[i];
                quadData[i][1] = hp[i] * hp[i]; // Yeni özellik
                quadData[i][2] = mpg[i];
            }
            DataFrame quadVeri = DataFrame.of(quadData, "hp", "hp2", "mpg");
            LinearModel quadModel = OLS.fit(formula, quadVeri);

            double quadB0 = quadModel.intercept();
            double quadB1 = quadModel.coefficients().get(0); // hp katsayısı
            double quadB2 = quadModel.coefficients().get(1); // hp2 katsayısı

            System.out.println("\n\n----2. DERECE QUADRATIC MODEL----");
            System.out.printf("mpg= %.4f + (%.4f) * hp + (%.4f) * hp*hp %n", quadB0, quadB1, quadB2);


            // 3. DERECE CUBIC MODEL
            double[][] cubicData = new double[n][4];
            for (int i = 0; i < n; i++) {
                cubicData[i][0] = hp[i];
                cubicData[i][1] = hp[i] * hp[i];
                cubicData[i][2] = hp[i] * hp[i] * hp[i]; // Yeni özellik
                cubicData[i][3] = mpg[i];
            }
            DataFrame cubicVeri = DataFrame.of(cubicData, "hp", "hp2", "hp3", "mpg");
            LinearModel cubicModel = OLS.fit(formula, cubicVeri);

            double cubicB0 = cubicModel.intercept();
            double cubicB1 = cubicModel.coefficients().get(0); // hp katsayısı
            double cubicB2 = cubicModel.coefficients().get(1); // hp2 katsayısı
            double cubicB3 = cubicModel.coefficients().get(2); // hp3 katsayısı

            System.out.println("\n\n----3. DERECE QUADRATIC MODEL----");
            System.out.printf("mpg= %.4f + (%.4f) * hp + (%.4f) * hp*hp + (%.4f) * hp*hp*hp%n", cubicB0, cubicB1, cubicB2, cubicB3);

            System.out.println("MODELLERİN SONUÇLARI: ");
            System.out.println(lineerModel);
            System.out.println(quadModel);
            System.out.println(cubicModel);

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
