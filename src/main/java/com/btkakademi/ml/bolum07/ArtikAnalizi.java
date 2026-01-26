package com.btkakademi.ml.bolum07;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LinearModel;
import smile.regression.OLS;

import java.nio.file.Path;
import java.util.Arrays;

public class ArtikAnalizi {
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
            int n = veriSeti.nrow();
            String hedefSutun = "mpg";

            int hedefIndex = 0;
            for (int i = 0; i < veriSeti.ncol(); i++) {
                if (veriSeti.column(i).name().equalsIgnoreCase(hedefSutun)) {
                    hedefIndex = i;
                    break;
                }
            }

            // Modeli Eğit
            Formula formula = Formula.lhs(hedefSutun);
            LinearModel model = OLS.fit(formula, veriSeti);

            // Artıkları Hesapla
            double[] artiklar = new double[n];

            double[] gercekDegerler = new double[n];
            double[] tahminDegerler = new double[n];

            double artikToplam = 0;
            double artikKareToplam = 0;
            double gercekToplam = 0;

            for (int i = 0; i < n; i++) {
                gercekDegerler[i] = veriSeti.getDouble(i, hedefIndex);
                tahminDegerler[i] = model.predict(veriSeti.get(i));

                artiklar[i] = gercekDegerler[i] - tahminDegerler[i];

                artikToplam += artiklar[i];
                artikKareToplam += artiklar[i] * artiklar[i];
                gercekToplam += gercekDegerler[i];
            }

            // Temel İstatistikler
            System.out.println("Temel İstatistikler");

            double gercekOrtalama = gercekToplam / n;
            double ssTot = 0;
            for (double gercek : gercekDegerler) {
                ssTot += Math.pow(gercek - gercekOrtalama, 2);
            }

            double r2 = 1 - (artikKareToplam / ssTot);
            System.out.println("R2 değeri: " + r2);

            double artikOrtalama = artikToplam / n;
            System.out.println("Artık Ortalaması: " + artikOrtalama);

            double artikVaryans = (artikKareToplam / n) - (artikOrtalama * artikOrtalama);
            double artikStdSapma = Math.sqrt(artikVaryans);
            System.out.println("Artık Std Sapma: " + artikStdSapma);

            double[] siraliArtiklar = artiklar.clone();
            Arrays.sort(siraliArtiklar);

            double artikMin = siraliArtiklar[0];
            double artikMedyan = siraliArtiklar[n / 2];
            double artikMax = siraliArtiklar[n - 1];

            System.out.printf("Min= %+.2f, Medyan= %+.2f, Max= %+.2f%n", artikMin, artikMedyan, artikMax);

            // Z-Score Analizi
            System.out.println("\nZ-Score Analizi\n");
            int z2Icinde = 0;
            int outlier = 0;

            for (double artik : artiklar) {
                double z = (artik - artikOrtalama) / artikStdSapma;
                double absZ = Math.abs(z);

                if (absZ < 2) z2Icinde++;
                if (absZ >= 3) outlier++;
            }

            double z2Yuzde = z2Icinde * 100.0 / n;
            double outlierYuzde = outlier * 100.0 / n;

            System.out.println("|z| < 2 içinde şu kadar örnek var: " + z2Icinde + " ve yuzdesi: " + z2Yuzde);
            System.out.println("|z| >= 3 içinde şu kadar örnek var: " + outlier + " ve yuzdesi: " + outlierYuzde);

            if (z2Yuzde >= 94 && z2Yuzde <= 97) {
                System.out.println("Dagilim normale cok yakindir");
            } else {
                System.out.println("Dagilimda sapma mevcut");
            }

            // Normallik Testi
            double skewnessToplam = 0;
            double kurtosisToplam = 0;

            for (double artik : artiklar) {
                double z = (artik - artikOrtalama) / artikStdSapma;
                skewnessToplam += Math.pow(z, 3);
                kurtosisToplam += Math.pow(z, 4);
            }

            double skewness = skewnessToplam / n;
            double kurtosis = kurtosisToplam / n;
            double excessKurtosis = kurtosis - 3; // Normal dağılım için 0 gelmesi gerekir.

            System.out.println("Skewness (Çarpıklık): " + skewness);
            if (Math.abs(skewness) < 0.5) {
                System.out.println("Simetriktir.");
            } else if (skewness > 0) {
                System.out.println("Sağa çarpıktır.");
            } else {
                System.out.println("Sola çarpıktır");
            }

            System.out.println("Kurtosis (Basıklık): " + kurtosis);
            System.out.println("Excess Kurtosis: " + excessKurtosis);
            if (Math.abs(excessKurtosis) < 1) {
                System.out.println("Normal dağılıma yakındır.");
            } else if (excessKurtosis > 0) {
                System.out.println("Sivri dağılım söz konusudur.");
            } else {
                System.out.println("Basık dağılım");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
