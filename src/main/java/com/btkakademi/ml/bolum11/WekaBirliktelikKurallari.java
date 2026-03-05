package com.btkakademi.ml.bolum11;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.InputStream;
import java.util.List;

public class WekaBirliktelikKurallari {
    static void main() {
        try {
            InputStream is = WekaBirliktelikKurallari.class.getClassLoader().getResourceAsStream("datasets/mushroom.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);
            Instances veri = loader.getDataSet();

            // Apriori
            Apriori apriori = new Apriori();
            apriori.setNumRules(10);
            apriori.buildAssociations(veri);
            System.out.println(apriori);

            // Farklı Confidence Eşikleri
            Apriori aprioriConf = new Apriori();
            aprioriConf.setNumRules(500);
            aprioriConf.setMinMetric(0.5);
            aprioriConf.setUpperBoundMinSupport(0.5);
            aprioriConf.setLowerBoundMinSupport(0.3);
            aprioriConf.buildAssociations(veri);

            List<AssociationRule> tumKurallar = aprioriConf.getAssociationRules().getRules();
            System.out.println("Toplam bulunan kural: " + tumKurallar.size());

            // Confidence dağılımı
            double minConf = 1.0, maxConf = 0.0, toplamConf = 0.0;
            for (AssociationRule kural : tumKurallar) {
                double c = kural.getPrimaryMetricValue();
                if (c < minConf) minConf = c;
                if (c > maxConf) maxConf = c;
                toplamConf += c;
            }

            System.out.printf("Confidence dağılımı: min=%.2f, max=%.2f, ort=%.2f", minConf, maxConf, toplamConf / tumKurallar.size());

            double[] confidenceEsikleri = {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0};
            for (double esik : confidenceEsikleri) {
                int sayi = 0;
                for (AssociationRule kural : tumKurallar) {
                    if (kural.getPrimaryMetricValue() >= esik) {
                        sayi++;
                    }
                }

                System.out.printf("  Confidence >= %3.0f%%: %3d / %d kural\n",
                        esik * 100, sayi, tumKurallar.size());
            }

            // Farklı Support Eşikleri
            double[] supportDegerleri = {0.3, 0.5, 0.7, 0.9};
            for (double sup : supportDegerleri) {
                try {
                    Apriori aprioriSup = new Apriori();
                    aprioriSup.setNumRules(10);
                    aprioriSup.setMinMetric(0.8);

                    aprioriSup.setUpperBoundMinSupport(sup);
                    aprioriSup.setLowerBoundMinSupport(Math.max(0.25, sup - 0.05));

                    aprioriSup.buildAssociations(veri);

                    List<AssociationRule> supKurallar = aprioriSup.getAssociationRules().getRules();
                    System.out.printf("\nSupport >= %.0f%%, Confidence >= 80%%: %d kural bulundu.\n"
                            , sup * 100, supKurallar.size());

                    int gosterilecek = Math.min(3, supKurallar.size());
                    for (int i = 0; i < gosterilecek; i++) {
                        System.out.printf("  %d. %s\n", i + 1, supKurallar.get(i));
                    }
                    if (supKurallar.size() > gosterilecek) {
                        System.out.println(" ...");
                    }

                } catch (OutOfMemoryError e) {
                    System.out.println("Hafiza hatasi.");
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
