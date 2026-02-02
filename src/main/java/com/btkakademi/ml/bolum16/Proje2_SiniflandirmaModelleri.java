package com.btkakademi.ml.bolum16;

import smile.classification.GradientTreeBoost;
import smile.classification.KNN;
import smile.classification.LogisticRegression;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.model.cart.SplitRule;
import smile.validation.metric.Accuracy;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class Proje2_SiniflandirmaModelleri {

    private static final String[] FN = {
            "Yas", "Cinsiyet", "Gogus_TA", "Gogus_ATA", "Gogus_NAP", "Gogus_ASY",
            "DKB", "Kolesterol", "AKS", "EKG_Normal", "EKG_ST", "EKG_LVH",
            "MKH", "Angina", "Oldpeak", "ST_Egim"
    };

    public static void main(String[] args) {
        try {
            System.out.println("=== PROJE 2: KALP HASTALIĞI RİSK TAHMİN SİSTEMİ ===");
            System.out.println("=== DOSYA 2/3: SINIFLANDIRMA MODELLERİ ===\n");

            // ============ 1. VERİ HAZIRLAMA ============
            System.out.println("============ 1. VERİ HAZIRLAMA ============\n");

            // Veri yükle + encode + normalize (Dosya 1'deki işlemlerin kompakt hali)
            var is = Proje2_SiniflandirmaModelleri.class.getClassLoader()
                    .getResourceAsStream("datasets/kalp-hastaligi.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            String[] bas = reader.readLine().split(",");
            Map<String, Integer> col = new java.util.LinkedHashMap<>();
            for (int i = 0; i < bas.length; i++) col.put(bas[i].trim(), i);
            List<String[]> satirlar = new ArrayList<>();
            String satir;
            while ((satir = reader.readLine()) != null)
                if (!satir.trim().isEmpty()) satirlar.add(satir.split(",", -1));
            reader.close();

            int n = satirlar.size(), F = 16, kI = col.get("Kolesterol"), bI = col.get("Dinlenme_Kan_Basinci");

            // Medyan hesapla + eksik doldur
            List<Double> kG = new ArrayList<>(), bG = new ArrayList<>();
            for (String[] p : satirlar) {
                double k = Double.parseDouble(p[kI].trim()), b = Double.parseDouble(p[bI].trim());
                if (k > 0) kG.add(k);
                if (b > 0) bG.add(b);
            }
            Collections.sort(kG);
            Collections.sort(bG);
            double kM = kG.get(kG.size() / 2), bM = bG.get(bG.size() / 2);
            for (String[] p : satirlar) {
                if (Double.parseDouble(p[kI].trim()) == 0) p[kI] = String.valueOf((int) kM);
                if (Double.parseDouble(p[bI].trim()) == 0) p[bI] = String.valueOf((int) bM);
            }

            // Encoding + hedef
            double[][] X = new double[n][F];
            int[] y = new int[n];
            for (int i = 0; i < n; i++) {
                String[] p = satirlar.get(i);
                X[i][0] = Double.parseDouble(p[col.get("Yas")].trim());
                X[i][1] = p[col.get("Cinsiyet")].trim().equals("E") ? 1 : 0;
                String g = p[col.get("Gogus_Agrisi_Tipi")].trim();
                X[i][2] = g.equals("TA") ? 1 : 0;
                X[i][3] = g.equals("ATA") ? 1 : 0;
                X[i][4] = g.equals("NAP") ? 1 : 0;
                X[i][5] = g.equals("ASY") ? 1 : 0;
                X[i][6] = Double.parseDouble(p[bI].trim());
                X[i][7] = Double.parseDouble(p[kI].trim());
                X[i][8] = Double.parseDouble(p[col.get("Aclik_Kan_Sekeri")].trim());
                String e = p[col.get("Dinlenme_EKG")].trim();
                X[i][9] = e.equals("Normal") ? 1 : 0;
                X[i][10] = e.equals("ST") ? 1 : 0;
                X[i][11] = e.equals("LVH") ? 1 : 0;
                X[i][12] = Double.parseDouble(p[col.get("Maks_Kalp_Hizi")].trim());
                X[i][13] = p[col.get("Egzersiz_Angina")].trim().equals("E") ? 1 : 0;
                X[i][14] = Double.parseDouble(p[col.get("ST_Oldpeak")].trim());
                String eg = p[col.get("ST_Egim")].trim();
                X[i][15] = eg.equals("Asagi") ? 0 : eg.equals("Duz") ? 1 : 2;
                y[i] = Integer.parseInt(p[bas.length - 1].trim());
            }

            // Min-Max normalizasyon (inline)
            double[] mn = new double[F], mx = new double[F];
            Arrays.fill(mn, Double.MAX_VALUE);
            Arrays.fill(mx, -Double.MAX_VALUE);
            for (double[] r : X)
                for (int j = 0; j < F; j++) {
                    if (r[j] < mn[j]) mn[j] = r[j];
                    if (r[j] > mx[j]) mx[j] = r[j];
                }
            double[][] XN = new double[n][F];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < F; j++) XN[i][j] = (X[i][j] - mn[j]) / (mx[j] - mn[j] + 1e-10);

            // Stratified split (inline)
            List<Integer> s0 = new ArrayList<>(), s1 = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == 0) s0.add(i);
                else s1.add(i);
            }
            Random rnd = new Random(42);
            Collections.shuffle(s0, rnd);
            Collections.shuffle(s1, rnd);
            int e0 = (int) (s0.size() * 0.7), e1 = (int) (s1.size() * 0.7);
            List<Integer> eg = new ArrayList<>(), te = new ArrayList<>();
            eg.addAll(s0.subList(0, e0));
            eg.addAll(s1.subList(0, e1));
            te.addAll(s0.subList(e0, s0.size()));
            te.addAll(s1.subList(e1, s1.size()));
            Collections.shuffle(eg, rnd);
            Collections.shuffle(te, rnd);

            double[][] XTr = new double[eg.size()][F];
            int[] yTr = new int[eg.size()];
            double[][] XTe = new double[te.size()][F];
            int[] yTe = new int[te.size()];
            for (int i = 0; i < eg.size(); i++) {
                XTr[i] = XN[eg.get(i)];
                yTr[i] = y[eg.get(i)];
            }
            for (int i = 0; i < te.size(); i++) {
                XTe[i] = XN[te.get(i)];
                yTe[i] = y[te.get(i)];
            }

            // DataFrame (RF, DT, GB için)
            Formula formula = Formula.lhs("Hedef");
            DataFrame trDf = df(XTr, yTr), teDf = df(XTe, yTe);

            System.out.printf("Veri: %d, Eğitim: %d, Test: %d\n\n", n, XTr.length, XTe.length);

            // ============ 2. KNN ============
            // K-Nearest Neighbors: Yeni noktayı K en yakın komşunun çoğunluk oyuyla sınıflar.
            // Lazy learner — eğitim aşaması yok, tahmin anında tüm veriyle mesafe hesaplar.
            // K küçük → overfitting (gürültüye duyarlı), K büyük → underfitting (bulanıklaşır)
            // Normalizasyon ZORUNLU — farklı ölçeklerde mesafe anlamsız olur.
            // Smile API: KNN.fit(X, y, k)
            System.out.println("============ 2. KNN ============\n");
            var knn = KNN.fit(XTr, yTr, 5); // K=5
            int[] knnP = new int[XTe.length];
            for (int i = 0; i < XTe.length; i++) knnP[i] = knn.predict(XTe[i]);
            metrik("KNN (K=5)", yTe, knnP);

            // ============ 3. KARAR AĞACI ============
            // If-else kuralları öğrenir. Her düğümde en iyi bölmeyi seçer (GINI/Entropy).
            // GINI safsızlık: 0=saf (tek sınıf), 0.5=en karışık (50-50)
            // Avantaj: Yorumlanabilir ("Yaş>50 VE Angina=1 → Hasta")
            // Dezavantaj: Overfitting eğilimi — pruning veya ensemble ile çözülür
            // Smile'da tek ağaç yok → ntrees=1 ile RF kullanılır
            System.out.println("\n============ 3. KARAR AĞACI ============\n");
            var dt = RandomForest.fit(formula, trDf,
                    new RandomForest.Options(1, 0, SplitRule.GINI, 10, 0, 5, 1.0, null, null, null));
            int[] dtP = new int[XTe.length];
            for (int i = 0; i < XTe.length; i++) dtP[i] = dt.predict(teDf.get(i));
            metrik("Karar Ağacı", yTe, dtP);

            // ============ 4. LOJİSTİK REGRESYON ============
            // P(hasta) = sigmoid(w₁x₁ + w₂x₂ + ... + b)
            // Sigmoid: 1/(1+e^(-z)) → her değeri [0,1] aralığına sıkıştırır
            // Eşik 0.5: P>0.5 → hasta tahmin, P<0.5 → sağlıklı tahmin
            // Avantaj: Olasılık çıktısı verir, katsayılar yorumlanabilir, hızlı
            // Dezavantaj: Sadece doğrusal sınır çizer — karmaşık ilişkilerde yetersiz
            // Smile API: LogisticRegression.fit(X, y)
            System.out.println("\n============ 4. LOJİSTİK REGRESYON ============\n");
            var lr = LogisticRegression.fit(XTr, yTr);
            int[] lrP = new int[XTe.length];
            for (int i = 0; i < XTe.length; i++) lrP[i] = lr.predict(XTe[i]);
            metrik("Lojistik Regresyon", yTe, lrP);

            // ============ 5. RANDOM FOREST ============
            // Birçok karar ağacının birleşimi (ensemble — bagging).
            // Her ağaç: Bootstrap örneklem + rastgele özellik alt kümesi (mtry)
            // Bootstrap: n elemandan n tane rastgele seç (tekrarlı) → ~%63'ü kullanılır
            // mtry=0 → sqrt(p) otomatik (sınıflandırmada standart)
            // Tahmin: Tüm ağaçların çoğunluk oyu
            // OOB (Out-of-Bag): Her ağacın görmediği ~%37 ile hata tahmini
            // Avantaj: Overfitting'e dirençli, özellik önemliliği verir
            // Smile API: RandomForest.fit(formula, df, Options)
            //   Options(ntrees, mtry, splitRule, maxDepth, maxNodes, nodeSize, subsample, ...)
            System.out.println("\n============ 5. RANDOM FOREST ============\n");
            var rf = RandomForest.fit(formula, trDf,
                    new RandomForest.Options(100, 0, SplitRule.GINI, 20, 0, 5, 1.0, null, null, null));
            int[] rfP = new int[XTe.length];
            for (int i = 0; i < XTe.length; i++) rfP[i] = rf.predict(teDf.get(i));
            metrik("Random Forest", yTe, rfP);

            // Özellik önemliliği (Top 5)
            double[] imp = rf.importance();
            Integer[] ii = new Integer[imp.length];
            for (int i = 0; i < imp.length; i++) ii[i] = i;
            Arrays.sort(ii, (a, b) -> Double.compare(imp[b], imp[a]));
            for (int i = 0; i < 5; i++) System.out.printf("  %d. %-12s: %.4f\n", i + 1, FN[ii[i]], imp[ii[i]]);

            // ============ 6. GRADIENT BOOSTING ============
            // Ardışık ağaçlar, her yeni ağaç öncekinin HATALARINI düzeltir.
            // Hata = gerçek - tahmin (residuals / kalıntılar)
            // shrinkage (öğrenme hızı): Her ağacın katkı oranı (0.01-0.3)
            //   Küçük → yavaş ama stabil, büyük → hızlı ama aşırı düzeltme riski
            // maxDepth SIĞ tutulur (3-8) → her ağaç "zayıf öğrenici" (weak learner)
            // Smile API: GradientTreeBoost.fit(formula, df, Options)
            //   Options(ntrees, maxDepth, maxNodes, nodeSize, shrinkage, subsample, ...)
            System.out.println("\n============ 6. GRADIENT BOOSTING ============\n");
            var gb = GradientTreeBoost.fit(formula, trDf,
                    new GradientTreeBoost.Options(100, 6, 32, 5, 0.1, 0.8, null, null));
            int[] gbP = new int[XTe.length];
            for (int i = 0; i < XTe.length; i++) gbP[i] = gb.predict(teDf.get(i));
            metrik("Gradient Boosting", yTe, gbP);

            // ============ 7. ADABOOST (WEKA) ============
            // Adaptive Boosting: Yanlış sınıflandırılan örneklerin ağırlığını artırır.
            // Sonraki zayıf öğrenici bu "zor" örneklere odaklanır.
            // DecisionStump: Tek split'li karar ağacı (en basit weak learner).
            // Final: Tüm weak learner'ların ağırlıklı oyu.
            // Weka Instances formatı gerekir — double[][] → Instances dönüşümü yapılır.
            System.out.println("\n============ 7. ADABOOST (WEKA) ============\n");
            var ab = new AdaBoostM1();
            ab.setNumIterations(50);
            ab.setClassifier(new DecisionStump());
            Instances wTr = weka(XTr, yTr, "tr"), wTe = weka(XTe, yTe, "te");
            ab.buildClassifier(wTr);
            int[] abP = new int[XTe.length];
            for (int i = 0; i < XTe.length; i++) abP[i] = (int) ab.classifyInstance(wTe.instance(i));
            metrik("AdaBoost", yTe, abP);

            System.out.println("\n=== SINIFLANDIRMA MODELLERİ TAMAMLANDI ===");

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Metrik yazdır (Accuracy, Precision, Recall, F1).
     */
    private static void metrik(String isim, int[] g, int[] t) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < g.length; i++) {
            if (t[i] == 1 && g[i] == 1) tp++;
            else if (t[i] == 1 && g[i] == 0) fp++;
            else if (t[i] == 0 && g[i] == 1) fn++;
        }
        double p = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0;
        double r = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0;
        double f = (p + r) > 0 ? 2 * p * r / (p + r) : 0;
        System.out.printf("%-20s: Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f\n",
                isim, Accuracy.of(g, t), p, r, f);
    }

    /**
     * DataFrame oluştur (Smile).
     */
    private static DataFrame df(double[][] X, int[] y) {
        int n = X.length, m = X[0].length;
        double[][] v = new double[n][m + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(X[i], 0, v[i], 0, m);
            v[i][m] = y[i];
        }
        String[] nm = new String[m + 1];
        System.arraycopy(FN, 0, nm, 0, m);
        nm[m] = "Hedef";
        return DataFrame.of(v, nm).factorize("Hedef");
    }

    /**
     * Weka Instances oluştur.
     */
    private static Instances weka(double[][] X, int[] y, String name) {
        int m = X[0].length;
        var attrs = new ArrayList<Attribute>();
        for (String f : FN) attrs.add(new Attribute(f));
        attrs.add(new Attribute("Hedef", new ArrayList<>(List.of("0", "1"))));
        var inst = new Instances(name, attrs, X.length);
        inst.setClassIndex(m);
        for (int i = 0; i < X.length; i++) {
            double[] d = new double[m + 1];
            System.arraycopy(X[i], 0, d, 0, m);
            d[m] = y[i];
            inst.add(new DenseInstance(1.0, d));
        }
        return inst;
    }
}
