package com.btkakademi.ml.bolum06;

import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaKararAgaciRegresyon {
    static void main() {
        try {
            String dosyaYolu = "src/main/resources/datasets/boston-housing.arff";
            DataSource kaynak = new DataSource(dosyaYolu);
            Instances veriSeti = kaynak.getDataSet();

            veriSeti.setClassIndex(veriSeti.numAttributes() - 1);

            System.out.println("Veri Seti: " + veriSeti.numInstances());

            //REPTree Ağacı Modeli
            REPTree model = new REPTree();
            //model.setMinNum(3); Minimum yaprak boyutu
            //model.setMaxDepth(10); Maksimum derinlik

            // Modeli Eğit
            model.buildClassifier(veriSeti);

            // Karar Ağacını Ekranda Görme
            System.out.println(model.toString());

            // Tahmin Örnekleri
            System.out.println("Tahmin Örnekleri: ");
            for (int i = 0; i < Math.min(10, veriSeti.numAttributes()); i++) {
                Instance ornek = veriSeti.instance(i);
                double gercek = ornek.classValue();
                // classifyInstance() Tahmin Yapmak için
                double tahmin = model.classifyInstance(ornek);
                double hata = gercek - tahmin;
                System.out.println(" " + gercek + " " + tahmin + " " + hata);
            }

            // Model Performansı
            double toplamKareHata = 0;
            double toplamGercek = 0;
            double toplamGercekKare = 0;

            for (int i = 0; i < veriSeti.numInstances(); i++) {
                Instance ornek = veriSeti.instance(i);
                double gercek = ornek.classValue();
                // classifyInstance() Tahmin Yapmak için
                double tahmin = model.classifyInstance(ornek);
                double hata = gercek - tahmin;

                toplamKareHata += hata * hata;
                toplamGercek += gercek;
                toplamGercekKare += gercek * gercek;
            }

            // RMSE
            double rmse = Math.sqrt(toplamKareHata / veriSeti.numInstances());

            double ortalama = toplamGercek / veriSeti.numInstances();
            double ssTot = toplamGercekKare - (toplamGercek * toplamGercek / veriSeti.numInstances());
            double ssRes = toplamKareHata;
            double r2 = 1 - (ssRes / ssTot);


            System.out.println("\nModel Performansı\n");
            System.out.println("R2: " + r2);
            System.out.println("RMSE: " + rmse);

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
