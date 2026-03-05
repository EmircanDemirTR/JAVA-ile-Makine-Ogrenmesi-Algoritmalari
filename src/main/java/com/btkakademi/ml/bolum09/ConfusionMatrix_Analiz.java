package com.btkakademi.ml.bolum09;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class ConfusionMatrix_Analiz {
    static void main() {
        try {
            var is = ConfusionMatrix_Analiz.class.getClassLoader().getResourceAsStream("datasets/dengesizMetrikler.csv");
            var reader = new BufferedReader(new InputStreamReader(is));

            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<String> etiketler = new ArrayList<>();

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                ozellikler.add(new double[]{
                        Double.parseDouble(p[0]), Double.parseDouble(p[1])
                });
                etiketler.add(p[2].trim());
            }
            reader.close();

            long pozitifSayisi = etiketler.stream().filter(e -> e.equals("pozitif")).count();
            long negatifSayisi = etiketler.stream().filter(e -> e.equals("negatif")).count();

            System.out.println("Veri Seti Dagilimi");
            System.out.printf("Negatif (Saglikli): %d (%.1f%%)\n", negatifSayisi, 100.0 * negatifSayisi / etiketler.size());
            System.out.printf("Pozitif (hasta): %d (%.1f%%)\n", pozitifSayisi, 100.0 * pozitifSayisi / etiketler.size());

            // Weka Instances Oluşturma

            ArrayList<Attribute> attributes = new ArrayList<>();

            attributes.add(new Attribute("ozellik1"));
            attributes.add(new Attribute("ozellik2"));

            ArrayList<String> sinifDegerleri = new ArrayList<>();
            sinifDegerleri.add("negatif"); // index 0
            sinifDegerleri.add("pozitif"); // index 1
            attributes.add(new Attribute("sinif", sinifDegerleri));

            Instances veri = new Instances("dengesiz", attributes, ozellikler.size());
            veri.setClassIndex(2);

            for (int i = 0; i < ozellikler.size(); i++) {
                double[] instance = new double[3];
                instance[0] = ozellikler.get(i)[0]; // ozellik1
                instance[1] = ozellikler.get(i)[1]; // ozellik2
                // Sinif degeri = negatif:0, pozitif:1
                instance[2] = etiketler.get(i).equals("negatif") ? 0 : 1;
                veri.add(new DenseInstance(1.0, instance));
            }


            // Doğruluk Paradoksu
            System.out.println("SENARYO 1: BASELINE MODEL");
            int baselineTP = 0;
            int baselineTN = (int) negatifSayisi;
            int baselineFP = 0;
            int baselineFN = (int) pozitifSayisi;

            System.out.println(baselineTP + " " + baselineFN);
            System.out.println(baselineFP + " " + baselineTN);

            double baselineAccuracy = 100.0 * (baselineTP + baselineTN) / (baselineFN + baselineFP + baselineTN + baselineTP);
            System.out.println("Dogruluk Orani: " + baselineAccuracy);


            // 2. SENARYO - GERÇEK MODEL
            J48 model = new J48();
            model.setConfidenceFactor(0.25F);
            model.buildClassifier(veri);

            Evaluation eval = new Evaluation(veri);
            eval.evaluateModel(model, veri);

            System.out.println(eval.toMatrixString());
            System.out.println("Dogruluk Orani: " + eval.pctCorrect());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
