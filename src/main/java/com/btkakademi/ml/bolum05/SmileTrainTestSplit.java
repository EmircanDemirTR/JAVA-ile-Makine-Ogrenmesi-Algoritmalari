package com.btkakademi.ml.bolum05;

import org.apache.commons.csv.CSVFormat;
import smile.classification.KNN;
import smile.io.Read;
import smile.validation.metric.Accuracy;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@SuppressWarnings("deprecation")
public class SmileTrainTestSplit {
    static void main() throws Exception {

        // CSV dosyasi yukleme
        var veri = Read.csv(veriYolu(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        int n = veri.nrow();
        System.out.println("Toplam veri: " + n);

        // 2. Sınıf etiketlerini sayiya donustur
        Map<String, Integer> etiketMap = new HashMap<>();
        etiketMap.put("setosa", 0);
        etiketMap.put("versicolor", 1);
        etiketMap.put("virginica", 2);

        //3. Ozellik Matrisi: X olsun. Hedef dizisi y olsun.
        // X: Her satir bir ornek, her sutun bir ozellik. (Toplam 4 ozellik)
        double[][] X = new double[n][4];
        int[] y = new int[n];


        for (int i = 0; i < n; i++) {
            X[i][0] = veri.getDouble(i, 0);
            X[i][1] = veri.getDouble(i, 1);
            X[i][2] = veri.getDouble(i, 2);
            X[i][3] = veri.getDouble(i, 3);
            y[i] = etiketMap.get(veri.get(i).getString("class"));
        }

        //4. Veriyi rastgele karıştır
        Random rand = new Random(42);
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            double[] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;

            int tempY = y[i];
            y[i] = y[j];
            y[j] = tempY;
        }

        //5. Veriyi egitim ve test olarak ayırma
        int egitimBoyut = (int) (n * 0.7);

        double[][] XEgitim = Arrays.copyOfRange(X, 0, egitimBoyut);
        int[] yEgitim = Arrays.copyOfRange(y, 0, egitimBoyut);
        double[][] XTest = Arrays.copyOfRange(X, egitimBoyut, n);
        int[] yTest = Arrays.copyOfRange(y, egitimBoyut, n);

        System.out.println("Egitim seti: " + egitimBoyut);
        System.out.println("Test seti: " + (n - egitimBoyut));

        //6. KNN modelini sadece egitim verisiyle egit
        var model = KNN.fit(XEgitim, yEgitim, 3);

        //7. Test verisi uzerinden toplu tahmin yapma
        int[] tahminler = model.predict(XTest);

        //8. Dogruluk hesabi ve ekrana yazdirma
        double dogruluk = Accuracy.of(yTest, tahminler);
        System.out.println("Dogruluk orani: " + dogruluk * 100);

    }

    private static Path veriYolu() throws URISyntaxException {
        var url = SmileModelEgitimi.class.getClassLoader().getResource("datasets/iris.csv");
        return Paths.get(url.toURI());
    }
}
