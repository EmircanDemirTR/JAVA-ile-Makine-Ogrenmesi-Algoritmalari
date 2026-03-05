package com.btkakademi.ml.bolum05;

import org.apache.commons.csv.CSVFormat;
import smile.classification.KNN;
import smile.io.Read;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class SmileModelEgitimi {
    public static void main(String[] args) throws Exception {
        var veri = Read.csv(veriYolu(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        System.out.println("Veri seti yuklendi: " + veri.nrow() + " ornek");

        // 2. Sinif etiketlerini sayilara donustur
        Map<String, Integer> etiketMap = new HashMap<>();
        etiketMap.put("setosa", 0);
        etiketMap.put("versicolor", 1);
        etiketMap.put("virginica", 2);

        // 3. Ozellik matrisi X ve hedef dizi Y oluşturma
        int n = veri.nrow();
        double[][] X = new double[n][4];
        int[] y = new int[n];

        for (int i = 0; i < n; i++) {
            X[i][0] = veri.getDouble(i, 0);
            X[i][1] = veri.getDouble(i, 1);
            X[i][2] = veri.getDouble(i, 2);
            X[i][3] = veri.getDouble(i, 3);
            y[i] = etiketMap.get(veri.get(i).getString("class"));
        }

        // 4. KNN modeli egit
        var model = KNN.fit(X, y, 3);

        // Tahmin Yapalım
        String[] etiketler = {"setosa", "verisolor", "virginica"};

        for (int i = 0; i < 5; i++) {
            int tahmin = model.predict(X[i]);
            System.out.println("Ornek " + (i + 1) + ": Gercek=" + etiketler[y[i]] + ", Tahmin=" + etiketler[tahmin]);
        }
    }

    private static Path veriYolu() throws URISyntaxException {
        var url = SmileModelEgitimi.class.getClassLoader().getResource("datasets/iris.csv");
        return Paths.get(url.toURI());
    }
}
