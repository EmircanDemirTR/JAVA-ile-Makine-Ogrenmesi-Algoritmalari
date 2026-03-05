package com.btkakademi.ml.bolum04;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmileKategorikVeri {
    public static void main(String[] args) throws Exception {
        List<String[]> veri = veriYukle(); // Veri yükle: renk, boyut, fiyat

        // 1. Orijinal Veri
        System.out.println("1. Orijinal Veri");
        System.out.println("Satir sayisi: " + veri.size());
        System.out.println("\nIlk 5 ornek: ");
        for (int i = 0; i < 5; i++) {
            System.out.println(" " + Arrays.toString(veri.get(i)));
        }

        // 2. Boyut için Ordinal Encoding
        System.out.println("2. Boyut için Ordinal Encoding");

        Map<String, Integer> boyutOrdinal = new LinkedHashMap<>();
        boyutOrdinal.put("kucuk", 0);
        boyutOrdinal.put("orta", 1);
        boyutOrdinal.put("buyuk", 2);

        System.out.println("Boyut kodlari: " + boyutOrdinal);


        // 3. Renk için One-Hot Encoding
        System.out.println("3. Renk için One-Hot Encoding");
        Map<String, Integer> renkOneHot = labelMap(veri, 0);
        System.out.println("Renk kategorileri: " + renkOneHot);

        // 4. Final Dönüşüm
        System.out.println("\n Final Asamasi");
        double[][] X = encode(veri, renkOneHot, boyutOrdinal);
        System.out.println("İlk 5 ornek: ");
        String[] renkler = renkOneHot.keySet().toArray(new String[0]);

        for (int i = 0; i < 5; i++) {
            String[] orijinal = veri.get(i);
            System.out.println(" " + Arrays.toString(orijinal) + " -> " + formatArray(X[i]));
        }
    }

    private static List<String[]> veriYukle() throws Exception {
        var is = SmileKategorikVeri.class.getClassLoader().getResourceAsStream("datasets/kategorik.csv");
        var reader = new BufferedReader(new InputStreamReader(is));
        reader.readLine();

        List<String[]> satirlar = new ArrayList<>();
        String satir;
        while ((satir = reader.readLine()) != null) {
            satirlar.add(satir.split(","));
        }
        reader.close();
        return satirlar;
    }

    private static Map<String, Integer> labelMap(List<String[]> veri, int col) {
        Map<String, Integer> map = new LinkedHashMap<>();
        for (String[] satir : veri) {
            map.putIfAbsent(satir[col], map.size());
        }
        return map;
    }

    private static double[][] encode(List<String[]> veri, Map<String, Integer> renkOneHot, Map<String, Integer> boyutOrdinal) {
        int cols = renkOneHot.size() + 2;
        double[][] X = new double[veri.size()][cols];

        for (int i = 0; i < veri.size(); i++) {
            String[] satir = veri.get(i);

            int renkIdx = renkOneHot.get(satir[0]);
            X[i][renkIdx] = 1;

            X[i][renkOneHot.size()] = boyutOrdinal.get(satir[1]);

            X[i][cols - 1] = Double.parseDouble(satir[2]);
        }
        return X;
    }

    private static String formatArray(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append((int) arr[i]);
            if (i < arr.length - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }
}
