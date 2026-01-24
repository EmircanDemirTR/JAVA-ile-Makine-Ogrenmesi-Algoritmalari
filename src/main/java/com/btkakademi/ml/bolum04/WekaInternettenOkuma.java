package com.btkakademi.ml.bolum04;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.net.URI;

public class WekaInternettenOkuma {
    public static void main(String[] args) {

        okuVeYazdir("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                true,
                "UCI Iris");

        okuVeYazdir("https://raw.githubusercontent.com/EmircanDemirTR/JAVA-ile-Makine-Ogrenmesi-Algoritmalari/refs/heads/main/src/main/resources/datasets/winequality-red.csv",
                false,
                "Github Wine Quality");

    }

    private static void okuVeYazdir(String url, boolean baslikYok, String isim) {
        try {
            System.out.println("İsmi: " + isim);
            System.out.println("URL: " + url);

            //CSV Yükleme
            CSVLoader loader = new CSVLoader();
            loader.setSource(URI.create(url).toURL().openStream());
            loader.setNoHeaderRowPresent(baslikYok);


            // Veri Setini Al
            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Bilgileri Yazdir
            System.out.println("Ornek: " + veri.numInstances() + ", Ozellik: " + veri.numAttributes());
            System.out.println("Ilk ornek: " + veri.instance(0));

        } catch (Exception e) {
            System.out.println("Hata: " + e.getMessage());
        }


    }
}
