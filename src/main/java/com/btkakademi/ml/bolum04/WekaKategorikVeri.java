package com.btkakademi.ml.bolum04;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

import java.io.InputStream;
import java.util.ArrayList;

public class WekaKategorikVeri {
    public static void main() throws Exception {

        Instances veri = veriYukle();

        // 1. Orijinal Veri
        System.out.println("Orijinal Veri");
        System.out.println("Satir: " + veri.numInstances() + ", Ozellik: " + veri.numAttributes());
        System.out.println("\nOzellikler: ");
        for (int i = 0; i < veri.numAttributes(); i++) {
            var attr = veri.attribute(i);
            if (attr.isNominal()) {
                System.out.println(" " + attr.name() + ": " + attr.numValues() + " kategori");
            } else {
                System.out.println(" " + attr.name() + ": sayisal");
            }
        }

        System.out.println("\nIlk 5 ornek");
        for (int i = 0; i < 5; i++) {
            System.out.println(" " + veri.instance(i));
        }


        // 2. Boyut için Ordinal Encoding (kucuk=0, orta=1, buyuk=2)
        System.out.println("2. Ordinal Encoding");
        System.out.println("Mantik: kucuk:0 - orta:1 - buyuk:2");
        Instances ordinalVeri = boyutOrdinalEncode(veri);
        System.out.println("\nIlk 5 ornek: (boyut sayisal)");

        System.out.println("\nIlk 5 ornek");
        for (int i = 0; i < 5; i++) {
            System.out.println(" " + ordinalVeri.instance(i));
        }


        // 3. Renk için One-Hot Encoding
        System.out.println("\n3. Renk One-Hot Encoding");
        Instances finalVeri = renkOneHotEncode(ordinalVeri);

        System.out.println("Yeni ozellik sayisi: " + finalVeri.numAttributes());
        System.out.println("\nOzellik Listesi: ");
        for (int i = 0; i < finalVeri.numAttributes(); i++) {
            System.out.println(" " + i + ". " + finalVeri.attribute(i).name());
        }

        System.out.println("\nIlk 5 ornek: (Final)");

        System.out.println("\nIlk 5 ornek");
        for (int i = 0; i < 5; i++) {
            System.out.println(" " + finalVeri.instance(i));
        }


    }

    private static Instances veriYukle() throws Exception {
        InputStream is = WekaEksikVeriDoldurma.class.getClassLoader().getResourceAsStream("datasets/kategorik.csv");
        CSVLoader loader = new CSVLoader();
        loader.setSource(is);
        return loader.getDataSet();
    }

    private static Instances boyutOrdinalEncode(Instances veri) {
        ArrayList<Attribute> yeniAttrList = new ArrayList<>();

        yeniAttrList.add(veri.attribute(0)); // Renk aynı kalsın - nominal

        yeniAttrList.add(new Attribute("boyut_ordinal"));

        yeniAttrList.add(veri.attribute(2)); // fiyat ayni kalsin - sayisal

        Instances yeniVeri = new Instances("kategorik_ordinal", yeniAttrList, veri.numInstances()); // Yeni veri seti oluşturduk

        for (int i = 0; i < veri.numInstances(); i++) {
            // Verileri kopayalama ve boyutu dönüştürme
            double[] degerler = new double[3];

            degerler[0] = veri.instance(i).value(0);

            String boyut = veri.instance(i).stringValue(1);
            degerler[1] = switch (boyut) {
                case "kucuk" -> 0;
                case "orta" -> 1;
                case "buyuk" -> 2;
                default -> 1;
            };

            degerler[2] = veri.instance(i).value(2); // Fiyat ayni kalsin
            yeniVeri.add(new DenseInstance(1.0, degerler));
        }
        return yeniVeri;
    }

    private static Instances renkOneHotEncode(Instances veri) throws Exception {
        NominalToBinary filtre = new NominalToBinary();
        filtre.setInputFormat(veri);
        return Filter.useFilter(veri, filtre);
    }
}
