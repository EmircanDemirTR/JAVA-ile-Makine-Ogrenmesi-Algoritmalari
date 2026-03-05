package com.btkakademi.ml.bolum04;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;

import java.io.InputStream;

public class WekaVeriDengesizligi {
    public static void main(String[] args) throws Exception {

        // Orijinal Veri
        Instances veri = veriYukle();
        veri.setClassIndex(veri.numAttributes() - 1);

        System.out.println("Orijinal Veri: ");
        sinifDagilimi(veri);

        // Yöntem 1: Oversampling (Azinligi cogalt)
        System.out.println("Oversampling (Resample): ");
        Instances oversampled = oversample(veri);
        sinifDagilimi(oversampled);

        // Yöntem 2: SMOTE (sentetik veri üretmek için)
        System.out.println("SMOTE: ");
        Instances smoted = smote(veri);
        sinifDagilimi(smoted);

    }

    private static Instances veriYukle() throws Exception {
        InputStream is = WekaEksikVeriDoldurma.class.getClassLoader().getResourceAsStream("datasets/dengesiz.csv");
        CSVLoader loader = new CSVLoader();
        loader.setSource(is);
        return loader.getDataSet();
    }

    private static void sinifDagilimi(Instances veri) {
        int[] sayilar = new int[veri.numClasses()];

        for (int i = 0; i < veri.numInstances(); i++) {
            int sinif = (int) veri.instance(i).classValue();
            sayilar[sinif]++;
        }

        System.out.println("Toplam: " + veri.numInstances());
        for (int i = 0; i < veri.numClasses(); i++) {
            String sinifAdi = veri.classAttribute().value(i);
            double oran = (100.0 * sayilar[i]) / veri.numInstances();
            System.out.printf(" %s: %d (%.1f%%)%n", sinifAdi, sayilar[i], oran);
        }
    }

    private static Instances oversample(Instances veri) throws Exception {
        Resample resample = new Resample();
        resample.setBiasToUniformClass(1.0); // Eşit dağılım hedefleniyor
        resample.setInputFormat(veri);
        return Filter.useFilter(veri, resample);
    }

    private static Instances smote(Instances veri) throws Exception {
        SMOTE smote = new SMOTE();
        smote.setPercentage(200.0); // Azinligi %200 arttir.
        smote.setInputFormat(veri);
        return Filter.useFilter(veri, smote);
    }
}
