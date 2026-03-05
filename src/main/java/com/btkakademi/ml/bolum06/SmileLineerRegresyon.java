package com.btkakademi.ml.bolum06;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LinearModel;
import smile.regression.OLS;

import java.nio.file.Path;
import java.util.Arrays;

public class SmileLineerRegresyon {
    static void main() {
        try {

            String dosyaYolu = "src/main/resources/datasets/kariyer-maas.csv";

            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .build();

            DataFrame veriSeti = Read.csv(Path.of(dosyaYolu), format);

            System.out.println("Veri Seti: " + veriSeti.nrow());
            System.out.println("Sutunlar: " + Arrays.toString(veriSeti.names()));

            // Formül tanımlama

            Formula formula = Formula.lhs("maas");

            // Modeli Oluşturalım
            LinearModel model = OLS.fit(formula, veriSeti);

            // Katsayıları al
            double b0 = model.intercept();
            double b1 = model.coefficients().get(0);

            System.out.println("\nModel Katsayıları\n");
            System.out.printf("Kesişim (b0): %.4f%n", b0);
            System.out.printf("Eğim (b1): %.4f%n", b1);
            System.out.printf("Denklem: maas = %.4f + %.4f * deneyim_yili%n", b0, b1);


            // Tahmin Etme
            System.out.println("\nTahmin Örnekleri");
            double[] testDeneyimler = {0.0, 2.0, 5.0, 10.0};
            for (double deneyim : testDeneyimler) {
                double tahmin = b0 + b1 * deneyim;
                System.out.printf("Deneyim: %.1f yıl -> Tahmini maaş %.2f%n", deneyim, tahmin);
            }


            // Model Performansı
            double toplamKareHata = 0;
            for (int i = 0; i < veriSeti.nrow(); i++) {
                double gercek = veriSeti.getDouble(i, 1);
                double tahmin = model.predict(veriSeti.get(i));
                toplamKareHata += Math.pow(gercek - tahmin, 2);
            }

            double rmse = Math.sqrt(toplamKareHata / veriSeti.nrow());

            System.out.println("Model Performansı: ");
            System.out.printf("RMSE: %.4f%n", rmse);

            // SMILE MODEL ÖZETİ
            System.out.println("\n\nSMILE MODEL OZETİ\n\n");
            System.out.println(model);


        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
