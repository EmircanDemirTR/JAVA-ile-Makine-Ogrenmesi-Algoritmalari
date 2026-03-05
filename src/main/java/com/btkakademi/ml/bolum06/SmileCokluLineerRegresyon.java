package com.btkakademi.ml.bolum06;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LinearModel;
import smile.regression.OLS;

import java.nio.file.Path;

public class SmileCokluLineerRegresyon {
    static void main() {
        try {
            String dosyaYolu = "src/main/resources/datasets/auto-mpg.csv";

            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .build();


            DataFrame veriSeti = Read.csv(Path.of(dosyaYolu), format);

            System.out.println("Veri seti: " + veriSeti.nrow());
            System.out.println("Ozellik Sayisi: " + (veriSeti.ncol() - 1));

            String[] sutunlar = veriSeti.names();

            // Formül tanımlama
            Formula formula = Formula.lhs("mpg");

            // Modeli oluştur ve eğit
            LinearModel model = OLS.fit(formula, veriSeti);

            //Katsayıları inceleme
            double b0 = model.intercept();
            int katsayiSayisi = model.coefficients().size();

            System.out.println("\nModel Katsayıları\n");
            for (int i = 0; i < katsayiSayisi; i++) {
                String ozellik = sutunlar[i];
                double katsayi = model.coefficients().get(i);
                System.out.printf(" %s: %.4f%n", ozellik, katsayi);
            }
            System.out.println("Intercpt b0 = " + b0);


            // Tahmin etme
            System.out.println("\nTahmin Örnekleri\n");
            int hedefIndex = veriSeti.ncol() - 1;
            for (int i = 0; i < 5; i++) {
                double gercek = veriSeti.getDouble(i, hedefIndex);
                double tahmin = model.predict(veriSeti.get(i));
                System.out.printf(" Örnek: %d: Gerçek=%.1f mpg, Tahmin=%.2f mpg, Hata=%.2f%n", (i + 1), gercek, tahmin, (gercek - tahmin));
            }

            // Model Performansı
            double toplamKareHata = 0;
            double toplamMutlakHata = 0;

            for (int i = 0; i < veriSeti.nrow(); i++) {
                double gercek = veriSeti.getDouble(i, hedefIndex);
                double tahmin = model.predict(veriSeti.get(i));
                double hata = gercek - tahmin;

                toplamKareHata += hata * hata;
                toplamMutlakHata += Math.abs(hata);
            }

            int n = veriSeti.nrow();
            double rmse = Math.sqrt(toplamKareHata / n);
            double mae = toplamMutlakHata / n;

            System.out.println("\nModel Performansı\n");
            System.out.println("RMSE: " + rmse);
            System.out.println("MAE: " + mae);

            // SMILE MODEL ÖZETİ
            System.out.println("\n\n SMILE MODEL OZETİ\n\n");
            System.out.println(model);

        } catch (Exception e) {
            System.out.println("Hata " + e.getMessage());
            e.printStackTrace();
        }
    }
}
