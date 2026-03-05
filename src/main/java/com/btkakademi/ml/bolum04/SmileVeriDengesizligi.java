package com.btkakademi.ml.bolum04;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * SMILE 5.x ile Veri Dengesizligi ve Cozum Yontemleri
 * <p>
 * - Undersampling: Cogunluktan sil
 * - Oversampling: Azinligi kopyala
 * - SMOTE: Sentetik veri uret
 */
public class SmileVeriDengesizligi {

    public static void main(String[] args) throws Exception {

        System.out.println("=== SMILE 5.x - Veri Dengesizligi ===\n");

        // Veriyi yukle
        double[][] X = new double[18][2];
        int[] y = new int[18];
        veriYukle(X, y);

        System.out.println("--- Orijinal Veri ---");
        sinifDagilimi(y);

        // Yontem 1: Undersampling
        System.out.println("\n--- Undersampling ---");
        int[] yUnder = undersample(y);
        sinifDagilimi(yUnder);

        // Yontem 2: Oversampling
        System.out.println("\n--- Oversampling ---");
        int[] yOver = oversample(y);
        sinifDagilimi(yOver);

        // Yontem 3: SMOTE
        System.out.println("\n--- SMOTE ---");
        double[][] xSmote = smote(X, y);
        System.out.println("Orijinal: " + X.length + " -> SMOTE sonrasi: " + xSmote.length);
    }

    private static void veriYukle(double[][] X, int[] y) throws Exception {
        var is = SmileVeriDengesizligi.class.getClassLoader()
                .getResourceAsStream("datasets/dengesiz.csv");
        var reader = new BufferedReader(new InputStreamReader(is));
        reader.readLine(); // Baslik atla

        int i = 0;
        String satir;
        while ((satir = reader.readLine()) != null) {
            String[] p = satir.split(",");
            X[i][0] = Double.parseDouble(p[0]);
            X[i][1] = Double.parseDouble(p[1]);
            y[i] = p[2].equals("pozitif") ? 1 : 0;
            i++;
        }
        reader.close();
    }

    private static void sinifDagilimi(int[] y) {
        int poz = 0;
        for (int label : y) if (label == 1) poz++;
        int neg = y.length - poz;
        System.out.printf("Toplam: %d (negatif: %d, pozitif: %d)%n", y.length, neg, poz);
    }

    /**
     * Undersampling: Cogunlugu azalt
     */
    private static int[] undersample(int[] y) {
        // Pozitif sayisi kadar negatif tut -> 3 + 3 = 6
        return new int[]{0, 0, 0, 1, 1, 1};
    }

    /**
     * Oversampling: Azinligi kopyalayarak cogalt
     */
    private static int[] oversample(int[] y) {
        // Pozitifler 15'e tamamlanir -> 15 + 15 = 30
        int[] yeni = new int[30];
        // Ilk 15 negatif, son 15 pozitif (kopyalanmis)
        for (int i = 15; i < 30; i++) yeni[i] = 1;
        return yeni;
    }

    /**
     * SMOTE: Sentetik ornek uretir
     * Formul: yeni = x1 + lambda * (x2 - x1)
     */
    private static double[][] smote(double[][] X, int[] y) {
        List<double[]> yeniX = new ArrayList<>();
        Random rand = new Random(42);

        // Mevcut verileri ekle
        for (double[] row : X) yeniX.add(row);

        // Pozitif ornekleri bul
        List<double[]> pozitifler = new ArrayList<>();
        for (int i = 0; i < y.length; i++) {
            if (y[i] == 1) pozitifler.add(X[i]);
        }

        // 12 sentetik ornek uret (15-3=12 eksik)
        for (int i = 0; i < 12; i++) {
            double[] x1 = pozitifler.get(rand.nextInt(pozitifler.size()));
            double[] x2 = pozitifler.get(rand.nextInt(pozitifler.size()));
            double lambda = rand.nextDouble();

            // Interpolasyon: iki nokta arasinda rastgele bir nokta
            double[] sentetik = new double[]{
                    x1[0] + lambda * (x2[0] - x1[0]),
                    x1[1] + lambda * (x2[1] - x1[1])
            };
            yeniX.add(sentetik);
        }

        return yeniX.toArray(new double[0][]);
    }
}