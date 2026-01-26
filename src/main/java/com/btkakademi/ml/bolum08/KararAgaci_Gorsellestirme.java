package com.btkakademi.ml.bolum08;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class KararAgaci_Gorsellestirme {
    static void main() {
        try {
            var is = KararAgaci_Gorsellestirme.class.getClassLoader().getResourceAsStream("datasets/wine.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Model Eğitimi;
            veri.randomize(new Random(42));
            J48 tree = new J48();
            tree.setConfidenceFactor(0.25f);
            tree.buildClassifier(veri);

            // Ağaç Metrikleri
            System.out.println("Ağaç Metrikleri");
            System.out.printf("Toplam dugum: %.0f, Yaprak: %.0f%n", tree.measureTreeSize(), tree.measureNumLeaves());

            // Ağaç yapısı
            System.out.println("KARAR AGACI YAPISI");
            System.out.println(tree);

            // AĞACIN METİN ÇIKTISI
            String agacStr = tree.toString();
            String[] satirlar = agacStr.split("\n");

            System.out.println("KOK OZELLİK");
            for (String satir : satirlar) {
                if (satir.contains("<=") || satir.contains(">")) {
                    if (!satir.trim().startsWith("|")) {
                        String kokOzellik = satir.split("[<>=]")[0].trim();
                        System.out.println("En onemli ozellik: " + kokOzellik);
                        break;
                    }
                }
            }

            System.out.println("GRAPHVIZ DOT FORMATI");
            try {
                String dotFormat = tree.graph();

                if (dotFormat.length() > 600) {
                    System.out.println(dotFormat.substring(0, 600) + "\n devami var...");
                } else {
                    System.out.println(dotFormat);
                }

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
