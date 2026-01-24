package com.btkakademi.ml.bolum04;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

public class SmileInternettenOkuma {
    public static void main(String[] args) {

        okuIris("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
        okuWine("https://raw.githubusercontent.com/EmircanDemirTR/JAVA-ile-Makine-Ogrenmesi-Algoritmalari/refs/heads/main/src/main/resources/datasets/winequality-red.csv");

    }

    private static void okuIris(String url) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(URI.create(url).toURL().openStream()));

            List<double[]> ozellikler = new ArrayList<>();
            List<String> etiketler = new ArrayList<>();

            String satir;

            while ((satir = reader.readLine()) != null) {

                if (satir.trim().isEmpty()) continue;

                String[] p = satir.split(",");
                ozellikler.add(new double[]{
                        Double.parseDouble(p[0]),
                        Double.parseDouble(p[1]),
                        Double.parseDouble(p[2]),
                        Double.parseDouble(p[3]),
                });
                etiketler.add(p[4]);

            }

            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            System.out.println("Ornek: " + X.length + ", Ozellik: " + X[0].length);
            System.out.printf("Ilk Ornek: [%.1f, %.1f, %.1f, %.1f] -> %s", X[0][0], X[0][1], X[0][2], X[0][3], etiketler.get(0));
        } catch (Exception e) {
            System.out.println("Hata: " + e.getMessage());
        }
    }


    private static void okuWine(String url) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(URI.create(url).toURL().openStream()));

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> kaliteler = new ArrayList<>();

            String satir;
            boolean ilkSatir = true;

            while ((satir = reader.readLine()) != null) {

                if (ilkSatir) {
                    ilkSatir = false;
                    continue;
                }

                if (satir.trim().isEmpty()) continue;

                String[] p = satir.split(",");

                double[] oz = new double[11];
                for (int i = 0; i < 11; i++) {
                    oz[i] = Double.parseDouble(p[i]);
                }

                ozellikler.add(oz);
                kaliteler.add(Integer.parseInt(p[11]));

            }

            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            System.out.println("Ornek: " + X.length + ", Ozellik: " + X[0].length);
            System.out.printf("Ilk Ornek: [%.2f, %.2f, ...] -> Kalite: %d", X[0][0], X[0][1], kaliteler.get(0));


        } catch (Exception e) {
            System.out.println("Hata: " + e.getMessage());
        }
    }
}
