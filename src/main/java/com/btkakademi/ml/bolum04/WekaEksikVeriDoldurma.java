package com.btkakademi.ml.bolum04;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.InputStream;

public class WekaEksikVeriDoldurma {
    public static void main(String[] args) throws Exception {
        Instances veri = veriYukle();
        eksikAnalizi(veri);

        //Satir Silme Yontemi
        System.out.println("Satir Silme Yontemi Calisti: ");
        Instances silinen = satirSil(veriYukle());
        System.out.println("Onceki: " + veri.numInstances() + " satir");
        System.out.println("Sonraki: " + silinen.numInstances() + " satir");
        System.out.println("Silinen: " + (veri.numInstances() - silinen.numInstances()));

        // Ortalama/Mod ile doldurma yöntemi
        System.out.println("\nOrtalama ile Doldurma Yontemi Calisti");
        Instances doldurulan = ortalamaIleDoldur(veriYukle());
        eksikAnalizi(doldurulan);
        System.out.println("\nDoldurulan ilk 5 satir: ");
        for (int i = 0; i < 5; i++) {
            System.out.println(" " + doldurulan.instance(i));
        }
    }

    private static Instances veriYukle() throws Exception {
        InputStream is = WekaEksikVeriDoldurma.class.getClassLoader().getResourceAsStream("datasets/iris-missing.csv");
        CSVLoader loader = new CSVLoader();
        loader.setSource(is);
        return loader.getDataSet();
    }

    private static void eksikAnalizi(Instances veri) {
        System.out.println("Toplam Satir: " + veri.numInstances());

        // Özellik bazında eksik sayisi
        System.out.println("\nOzellik bazinda eksikler: ");

        for (int j = 0; j < veri.numAttributes(); j++) {
            int eksik = 0;
            for (int i = 0; i < veri.numInstances(); i++) {
                if (veri.instance(i).isMissing(j)) {
                    eksik++;
                }
            }
            if (eksik > 0) {
                System.out.println(" " + veri.attribute(j).name() + ": " + eksik);
            }
        }

        // Eksik içeren satir sayisi
        int eksikSatir = 0;
        for (int i = 0; i < veri.numInstances(); i++) {
            if (veri.instance(i).hasMissingValue()) {
                eksikSatir++;
            }
        }

        System.out.println("\nEksik iceren satir: " + eksikSatir);
    }

    private static Instances satirSil(Instances veri) {
        Instances temiz = new Instances(veri, 0);
        for (int i = 0; i < veri.numInstances(); i++) {
            if (!veri.instance(i).hasMissingValue()) {
                temiz.add(veri.instance(i));
            }
        }
        return temiz;
    }

    private static Instances ortalamaIleDoldur(Instances veri) throws Exception {
        ReplaceMissingValues filtre = new ReplaceMissingValues();
        filtre.setInputFormat(veri);
        return Filter.useFilter(veri, filtre);
    }
}
