package com.btkakademi.ml.bolum04;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.InputStream;

/**
 * WEKA ile Eksik Veri Tespiti ve Doldurma Yontemleri
 * <p>
 * Yontemler:
 * 1. Tespit - isMissing(), hasMissingValue()
 * 2. Satir Silme - eksik iceren satirlari cikar
 * 3. Ortalama/Mod - ReplaceMissingValues filtresi
 */
public class WekaEksikVeriDoldurma {

    public static void main(String[] args) throws Exception {

        System.out.println("=== WEKA - Eksik Veri ve Doldurma ===\n");

        // 1. Veriyi yukle ve eksikleri tespit et
        System.out.println("--- 1. Eksik Veri Tespiti ---");
        Instances veri = veriYukle();
        eksikAnalizi(veri);

        // 2. Satir silme yontemi
        System.out.println("\n--- 2. Yontem: Satir Silme ---");
        Instances silinen = satirSil(veriYukle());
        System.out.println("Onceki: " + veri.numInstances() + " satir");
        System.out.println("Sonraki: " + silinen.numInstances() + " satir");
        System.out.println("Silinen: " + (veri.numInstances() - silinen.numInstances()) + " satir");

        // 3. Ortalama/Mod ile doldurma
        System.out.println("\n--- 3. Yontem: Ortalama ile Doldurma ---");
        Instances doldurulan = ortalamaIleDoldur(veriYukle());
        eksikAnalizi(doldurulan);
        System.out.println("\nDoldurulan ilk 5 satir:");
        for (int i = 0; i < 5; i++) {
            System.out.println("  " + doldurulan.instance(i));
        }
    }

    /**
     * CSV dosyasini yukler
     */
    private static Instances veriYukle() throws Exception {
        InputStream is = WekaEksikVeriDoldurma.class.getClassLoader()
                .getResourceAsStream("datasets/iris-missing.csv");
        CSVLoader loader = new CSVLoader();
        loader.setSource(is);
        return loader.getDataSet();
    }

    /**
     * Eksik veri analizini yapar
     */
    private static void eksikAnalizi(Instances veri) {
        System.out.println("Toplam satir: " + veri.numInstances());

        // Ozellik bazinda eksik sayisi
        System.out.println("\nOzellik bazinda eksik:");
        for (int j = 0; j < veri.numAttributes(); j++) {
            int eksik = 0;
            for (int i = 0; i < veri.numInstances(); i++) {
                if (veri.instance(i).isMissing(j)) eksik++;
            }
            if (eksik > 0) {
                System.out.println("  " + veri.attribute(j).name() + ": " + eksik);
            }
        }

        // Eksik iceren satir sayisi
        int eksikSatir = 0;
        for (int i = 0; i < veri.numInstances(); i++) {
            if (veri.instance(i).hasMissingValue()) eksikSatir++;
        }
        System.out.println("\nEksik iceren satir: " + eksikSatir);
    }

    /**
     * Eksik iceren satirlari siler
     */
    private static Instances satirSil(Instances veri) {
        Instances temiz = new Instances(veri, 0);  // Bos kopya
        for (int i = 0; i < veri.numInstances(); i++) {
            if (!veri.instance(i).hasMissingValue()) {
                temiz.add(veri.instance(i));
            }
        }
        return temiz;
    }

    /**
     * Sayisal: Ortalama, Kategorik: Mod ile doldurur
     */
    private static Instances ortalamaIleDoldur(Instances veri) throws Exception {
        ReplaceMissingValues filtre = new ReplaceMissingValues();
        filtre.setInputFormat(veri);
        return Filter.useFilter(veri, filtre);
    }
}
