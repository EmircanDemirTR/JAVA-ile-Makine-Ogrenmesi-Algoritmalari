package com.btkakademi.ml.bolum04;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.InterquartileRange;

import java.io.InputStream;

public class WekaAykiriDeger {
    public static void main(String[] args) throws Exception {
        InputStream is = WekaAykiriDeger.class.getClassLoader().getResourceAsStream("datasets/aykiri.csv");

        CSVLoader loader = new CSVLoader();
        loader.setSource(is);
        Instances veri = loader.getDataSet();

        System.out.println("Toplam satir: " + veri.numInstances());

        // IQR Filtresi Uygulama
        InterquartileRange iqr = new InterquartileRange();
        iqr.setAttributeIndices("first-last");
        iqr.setOutlierFactor(1.5);
        iqr.setInputFormat(veri);
        Instances sonuc = Filter.useFilter(veri, iqr);

        // Aykiri Degerleri bul
        System.out.println("\nAykiri Degerler");
        int aykiriSayisi = 0;
        for (int i = 0; i < sonuc.numInstances(); i++) {
            if (sonuc.instance(i).stringValue(sonuc.numAttributes() - 1).equals("yes")) {
                System.out.println(" Satir " + i + ": " + veri.instance(i).value(0));
                aykiriSayisi++;
            }
        }
        System.out.println("\nToplam aykiri: " + aykiriSayisi);
    }
}
