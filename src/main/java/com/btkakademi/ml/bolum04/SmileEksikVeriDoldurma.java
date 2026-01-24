package com.btkakademi.ml.bolum04;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SmileEksikVeriDoldurma {
    public static void main(String[] args) throws Exception {

        // Veriyi yükle ve temel bilgileri göster
        double[][] veri = veriYukle();
        System.out.println("Toplam satir bilgisi: " + veri.length);
        System.out.println("Eksik iceren satir: " + eksikSayisi(veri) + "\n");

        // Yöntem 1: Satır silme
        System.out.println("Satir silme");
        double[][] silinen = satirSil(veri);
        System.out.println("Kalan satir: " + silinen.length);

        // Yöntem 2: Ortalama ile Doldurma - Manuel
        System.out.println("Ortalama ile Doldurma");
        double[][] ortDolu = ortalamaIleDoldurManuel(veriYukle());
        yazdir(ortDolu, 5);

        // Yöntem 3: Medyan ile Doldurma - STREAM API
        System.out.println("Medyan ile Doldurma");
        double[][] medDolu = medyanIleDoldurmaStream(veriYukle());
        yazdir(medDolu, 5);
    }

    private static double[][] veriYukle() throws Exception {
        var is = SmileEksikVeriDoldurma.class.getClassLoader().getResourceAsStream("datasets/iris-missing.csv");
        var reader = new BufferedReader(new InputStreamReader(is));

        // İlk satir başlık, atla
        reader.readLine();

        List<double[]> satirlar = new ArrayList<>();
        String satir;

        while ((satir = reader.readLine()) != null) {
            // split(","); -- boş alanları atlar
            // split (",", -1) -- boş alanları korur ve bunları "" olarak tutar
            String[] parcalar = satir.split(",", -1);

            double[] degerler = new double[4];
            for (int i = 0; i < 4; i++) {
                degerler[i] = parcalar[i].isEmpty() ? Double.NaN : Double.parseDouble(parcalar[i]);
            }
            satirlar.add(degerler);
        }
        reader.close();

        // Double ArrayList'i double[][] dönüştürerek döndürelim
        return satirlar.toArray(new double[0][]);
    }

    private static int eksikSayisi(double[][] X) {
        int sayac = 0;
        for (double[] satir : X) {
            for (double d : satir) {
                if (Double.isNaN(d)) {
                    sayac++;
                    break;
                }
            }
        }
        return sayac;
    }

    private static double[][] satirSil(double[][] X) {
        List<double[]> temiz = new ArrayList<>();

        for (double[] satir : X) {

            boolean eksikVar = Arrays.stream(satir).anyMatch(Double::isNaN);

            if (!eksikVar) {
                temiz.add(satir);
            }
        }
        return temiz.toArray(new double[0][]);
    }

    private static double[][] ortalamaIleDoldurManuel(double[][] X) {

        double[][] sonuc = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            sonuc[i] = Arrays.copyOf(X[i], X[i].length);
        }

        for (int j = 0; j < 4; j++) {
            double toplam = 0;
            int sayac = 0;

            for (int i = 0; i < sonuc.length; i++) {
                if (!Double.isNaN(sonuc[i][j])) {
                    toplam += sonuc[i][j];
                    sayac++;
                }
            }

            double ortalama = toplam / sayac;

            for (int i = 0; i < sonuc.length; i++) {

                if (Double.isNaN(sonuc[i][j])) {
                    sonuc[i][j] = ortalama;
                }
            }
        }
        return sonuc;
    }

    private static double[][] medyanIleDoldurmaStream(double[][] X) {

        double[][] sonuc = Arrays.stream(X).map(double[]::clone).toArray(double[][]::new);

        for (int j = 0; j < 4; j++) {
            final int col = j;

            double[] gecerli = Arrays.stream(sonuc).mapToDouble(satir -> satir[col])
                    .filter(d -> !Double.isNaN(d))
                    .sorted()
                    .toArray();

            double medyan = gecerli[gecerli.length / 2];

            for (double[] satir : sonuc) {
                if (Double.isNaN(satir[col])) {
                    satir[col] = medyan;
                }
            }
        }
        return sonuc;
    }

    private static void yazdir(double[][] X, int n) {
        for (int i = 0; i < Math.min(n, X.length); i++) {
            System.out.printf(" [%.2f, %.2f, %.2f, %.2f]%n", X[i][0], X[i][1], X[i][2], X[i][3]);
        }
    }

}
