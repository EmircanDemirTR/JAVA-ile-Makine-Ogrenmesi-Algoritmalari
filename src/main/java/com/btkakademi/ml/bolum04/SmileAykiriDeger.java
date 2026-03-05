package com.btkakademi.ml.bolum04;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SmileAykiriDeger {
    public static void main(String[] args) throws Exception {

        double[] veri = veriYukle();
        System.out.println("Toplam satir: " + veri.length);

        double[] sirali = Arrays.copyOf(veri, veri.length);
        Arrays.sort(sirali);

        double q1 = sirali[veri.length / 4]; // 25. yuzdelik
        double q3 = sirali[3 * veri.length / 4]; // 75. yuzdelik
        double iqr = q3 - q1;

        double altSinir = q1 - 1.5 * iqr;
        double ustSinir = q3 + 1.5 * iqr;

        // Aykiri Degerleri Bulalim
        System.out.println("\nAykiri Degerler: ");
        int aykiriSayisi = 0;
        for (int i = 0; i < veri.length; i++) {
            if (veri[i] < altSinir || veri[i] > ustSinir) {
                System.out.println(" Satir: " + i + (int) veri[i]);
                aykiriSayisi++;
            }
        }
        System.out.println("\nToplam aykiri: " + aykiriSayisi);
    }

    private static double[] veriYukle() throws Exception {
        var is = SmileAykiriDeger.class.getClassLoader().getResourceAsStream("datasets/aykiri.csv");
        var reader = new BufferedReader(new InputStreamReader(is));
        reader.readLine();

        List<Double> degerler = new ArrayList<>();
        String satir;
        while ((satir = reader.readLine()) != null) {
            degerler.add(Double.parseDouble(satir.trim()));
        }
        reader.close();
        return degerler.stream().mapToDouble(d -> d).toArray();
    }
}
