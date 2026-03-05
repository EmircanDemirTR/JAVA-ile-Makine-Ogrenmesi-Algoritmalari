package com.btkakademi.ml.bolum04;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.InputStream;

public class WekaOlcekleme {
    public static void main(String[] args) throws Exception {
        Instances veri = veriYukle();

        // 1. Orijinal Veri
        System.out.println("1. Orijinal Veri");
        System.out.println("Satir: " + veri.numInstances() + ", Özellik: " + veri.numAttributes());
        istatistikGoster(veri);
        for (int i = 0; i < 5; i++) {
            System.out.println(" " + veri.instance(i));
        }


        // 2. Min-Max Normalizasyon (0-1)
        System.out.println("2. Min-Max Normalizasyon");
        Instances minMax = minMaxNormalize(veri);
        istatistikGoster(minMax);
        System.out.println("\nIlk 5 ornek: ");
        for (int i = 0; i < 5; i++) {
            System.out.printf(" %.3f, %.3f, %.3f, %.3f%n", minMax.instance(i).value(0), minMax.instance(i).value(1),
                    minMax.instance(i).value(2), minMax.instance(i).value(3));
        }


        // 3. Z-Score Standardizasyon
        System.out.println("3. Z-Score Standardizasyon");
        Instances zScore = zScoreStandardize(veri);
        istatistikGoster(zScore);
        System.out.println("\nIlk 5 ornek: ");
        for (int i = 0; i < 5; i++) {
            System.out.printf(" %.3f, %.3f, %.3f, %.3f%n", zScore.instance(i).value(0), zScore.instance(i).value(1),
                    zScore.instance(i).value(2), zScore.instance(i).value(3));
        }

    }

    private static Instances veriYukle() throws Exception {
        InputStream is = WekaEksikVeriDoldurma.class.getClassLoader().getResourceAsStream("datasets/olcekleme.csv");
        CSVLoader loader = new CSVLoader();
        loader.setSource(is);
        return loader.getDataSet();
    }

    private static void istatistikGoster(Instances veri) {
        System.out.println("\nOzellik İstatistikleri: ");
        for (int i = 0; i < veri.numAttributes(); i++) {
            var stats = veri.attributeStats(i);
            System.out.printf(" %s: min=%.2f, max=%.2f, ort=%.2f%n", veri.attribute(i).name(), stats.numericStats.min, stats.numericStats.max, stats.numericStats.mean);
        }
    }

    private static Instances minMaxNormalize(Instances veri) throws Exception {
        Normalize filtre = new Normalize();
        filtre.setInputFormat(veri);
        return Filter.useFilter(veri, filtre);
    }

    private static Instances zScoreStandardize(Instances veri) throws Exception {
        Standardize filtre = new Standardize();
        filtre.setInputFormat(veri);
        return Filter.useFilter(veri, filtre);
    }
}
