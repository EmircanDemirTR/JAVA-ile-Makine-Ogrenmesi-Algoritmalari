package com.btkakademi.ml.bolum16;

import smile.classification.GradientTreeBoost;
import smile.classification.LogisticRegression;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.model.cart.SplitRule;
import smile.util.Index;
import smile.validation.Bag;
import smile.validation.CrossValidation;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Proje 2: Kalp Hastalığı Risk Tahmin Sistemi
 * Dosya 3/3: Model Değerlendirme ve İyileştirme
 * BTK Akademi - Java ile Makine Öğrenmesi
 */
public class Proje2_ModelDegerlendirmeVeIyilestirme {

    private static final String[] FN = {
            "Yas", "Cinsiyet", "Gogus_TA", "Gogus_ATA", "Gogus_NAP", "Gogus_ASY",
            "DKB", "Kolesterol", "AKS", "EKG_Normal", "EKG_ST", "EKG_LVH",
            "MKH", "Angina", "Oldpeak", "ST_Egim"
    };

    public static void main(String[] args) {
        try {
            System.out.println("=== PROJE 2: KALP HASTALIĞI RİSK TAHMİN SİSTEMİ ===");
            System.out.println("=== DOSYA 3/3: MODEL DEĞERLENDİRME VE İYİLEŞTİRME ===\n");

            // ============ 1. VERİ HAZIRLAMA ============
            System.out.println("============ 1. VERİ HAZIRLAMA ============\n");

            // Kompakt veri yükleme (encode + normalize + split)
            var is = Proje2_ModelDegerlendirmeVeIyilestirme.class.getClassLoader()
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
            List<Double> kG = new ArrayList<>(), bG = new ArrayList<>();
            for (String[] p : satirlar) {
                double k = Double.parseDouble(p[kI].trim()), b = Double.parseDouble(p[bI].trim());
                if (k > 0) kG.add(k); if (b > 0) bG.add(b);
            }
            Collections.sort(kG); Collections.sort(bG);
            double kM = kG.get(kG.size() / 2), bM = bG.get(bG.size() / 2);
            for (String[] p : satirlar) {
                if (Double.parseDouble(p[kI].trim()) == 0) p[kI] = String.valueOf((int) kM);
                if (Double.parseDouble(p[bI].trim()) == 0) p[bI] = String.valueOf((int) bM);
            }

            double[][] X = new double[n][F]; int[] y = new int[n];
            for (int i = 0; i < n; i++) {
                String[] p = satirlar.get(i);
                X[i][0] = Double.parseDouble(p[col.get("Yas")].trim());
                X[i][1] = p[col.get("Cinsiyet")].trim().equals("E") ? 1 : 0;
                String g = p[col.get("Gogus_Agrisi_Tipi")].trim();
                X[i][2] = g.equals("TA") ? 1 : 0; X[i][3] = g.equals("ATA") ? 1 : 0;
                X[i][4] = g.equals("NAP") ? 1 : 0; X[i][5] = g.equals("ASY") ? 1 : 0;
                X[i][6] = Double.parseDouble(p[bI].trim()); X[i][7] = Double.parseDouble(p[kI].trim());
                X[i][8] = Double.parseDouble(p[col.get("Aclik_Kan_Sekeri")].trim());
                String e = p[col.get("Dinlenme_EKG")].trim();
                X[i][9] = e.equals("Normal") ? 1 : 0; X[i][10] = e.equals("ST") ? 1 : 0; X[i][11] = e.equals("LVH") ? 1 : 0;
                X[i][12] = Double.parseDouble(p[col.get("Maks_Kalp_Hizi")].trim());
                X[i][13] = p[col.get("Egzersiz_Angina")].trim().equals("E") ? 1 : 0;
                X[i][14] = Double.parseDouble(p[col.get("ST_Oldpeak")].trim());
                String eg = p[col.get("ST_Egim")].trim();
                X[i][15] = eg.equals("Asagi") ? 0 : eg.equals("Duz") ? 1 : 2;
                y[i] = Integer.parseInt(p[bas.length - 1].trim());
            }

            // Normalize
            double[] mn = new double[F], mx = new double[F];
            Arrays.fill(mn, Double.MAX_VALUE); Arrays.fill(mx, -Double.MAX_VALUE);
            for (double[] r : X) for (int j = 0; j < F; j++) { if (r[j] < mn[j]) mn[j] = r[j]; if (r[j] > mx[j]) mx[j] = r[j]; }
            double[][] XN = new double[n][F];
            for (int i = 0; i < n; i++) for (int j = 0; j < F; j++) XN[i][j] = (X[i][j] - mn[j]) / (mx[j] - mn[j] + 1e-10);

            // Split
            List<Integer> s0 = new ArrayList<>(), s1 = new ArrayList<>();
            for (int i = 0; i < n; i++) { if (y[i] == 0) s0.add(i); else s1.add(i); }
            Random rnd = new Random(42);
            Collections.shuffle(s0, rnd); Collections.shuffle(s1, rnd);
            int e0 = (int) (s0.size() * 0.7), e1 = (int) (s1.size() * 0.7);
            List<Integer> egL = new ArrayList<>(), teL = new ArrayList<>();
            egL.addAll(s0.subList(0, e0)); egL.addAll(s1.subList(0, e1));
            teL.addAll(s0.subList(e0, s0.size())); teL.addAll(s1.subList(e1, s1.size()));
            Collections.shuffle(egL, rnd); Collections.shuffle(teL, rnd);

            double[][] XTr = new double[egL.size()][F]; int[] yTr = new int[egL.size()];
            double[][] XTe = new double[teL.size()][F]; int[] yTe = new int[teL.size()];
            for (int i = 0; i < egL.size(); i++) { XTr[i] = XN[egL.get(i)]; yTr[i] = y[egL.get(i)]; }
            for (int i = 0; i < teL.size(); i++) { XTe[i] = XN[teL.get(i)]; yTe[i] = y[teL.get(i)]; }

            Formula formula = Formula.lhs("Hedef");
            DataFrame trDf = df(XTr, yTr), teDf = df(XTe, yTe), tumDf = df(XN, y);
            System.out.printf("Toplam: %d, Eğitim: %d, Test: %d\n", n, XTr.length, XTe.length);

            // ============ 2. 10-FOLD CROSS-VALIDATION ============
            // CV: Veriyi K parçaya (fold) böler.
            // Her turda 1 parça test, K-1 parça eğitim → K farklı sonuç.
            // Ortalaması → gerçek performans tahmini, std → stabilite.
            //
            // Neden tek split yetmez?
            //   Tek split'te "şanslı" veya "şanssız" bölünme olabilir.
            //   10 farklı bölünmenin ortalaması daha güvenilir.
            //
            // Stratified CV: Her fold'da sınıf oranları korunur.
            //   Smile API: CrossValidation.stratify(y, k) → Bag[]
            //   Bag.samples() → eğitim indeksleri
            //   Bag.oob() → test indeksleri (out-of-bag)
            System.out.println("\n============ 2. 10-FOLD CROSS-VALIDATION ============\n");

            int KF = 10;
            String[] mIsim = {"Random Forest", "Gradient Boosting", "Lojistik Regresyon"};
            double[] cvOrt = new double[3], cvStd = new double[3];

            for (int mi = 0; mi < 3; mi++) {
                double[] accs = new double[KF];
                Bag[] folds = CrossValidation.stratify(y, KF);
                for (int f = 0; f < KF; f++) {
                    int[] fTrI = folds[f].samples(), fTeI = folds[f].oob();
                    if (mi < 2) { // RF veya GB → DataFrame
                        DataFrame fTr = tumDf.get(Index.of(fTrI)), fTe = tumDf.get(Index.of(fTeI));
                        var model = mi == 0
                                ? RandomForest.fit(formula, fTr, new RandomForest.Options(100, 0, SplitRule.GINI, 20, 0, 5, 1.0, null, null, null))
                                : GradientTreeBoost.fit(formula, fTr, new GradientTreeBoost.Options(100, 6, 32, 5, 0.1, 0.8, null, null));
                        int[] gercek = formula.y(fTe).toIntArray(); int d = 0;
                        for (int i = 0; i < fTe.nrow(); i++) if (model.predict(fTe.get(i)) == gercek[i]) d++;
                        accs[f] = (double) d / fTe.nrow();
                    } else { // LR → double[][]
                        double[][] fX1 = new double[fTrI.length][]; int[] fY1 = new int[fTrI.length];
                        double[][] fX2 = new double[fTeI.length][]; int[] fY2 = new int[fTeI.length];
                        for (int i = 0; i < fTrI.length; i++) { fX1[i] = XN[fTrI[i]]; fY1[i] = y[fTrI[i]]; }
                        for (int i = 0; i < fTeI.length; i++) { fX2[i] = XN[fTeI[i]]; fY2[i] = y[fTeI[i]]; }
                        var lrM = LogisticRegression.fit(fX1, fY1); int d = 0;
                        for (int i = 0; i < fX2.length; i++) if (lrM.predict(fX2[i]) == fY2[i]) d++;
                        accs[f] = (double) d / fX2.length;
                    }
                }
                double top = 0; for (double v : accs) top += v; cvOrt[mi] = top / KF;
                double st = 0; for (double v : accs) st += (v - cvOrt[mi]) * (v - cvOrt[mi]);
                cvStd[mi] = Math.sqrt(st / KF);
                System.out.printf("%-22s: CV = %.4f ± %.4f\n", mIsim[mi], cvOrt[mi], cvStd[mi]);
            }

            // ============ 3. ROC/AUC ============
            // ROC eğrisi: Eşik değiştikçe TPR (Sensitivity) vs FPR grafiği.
            // AUC = ROC altındaki alan (trapez yöntemi ile hesaplanır).
            //   1.0 → mükemmel ayrım, 0.5 → yazı-tura (rastgele)
            //   0.9+ → mükemmel, 0.8-0.9 → iyi, 0.7-0.8 → kabul edilebilir
            //
            // Neden Accuracy yetmez?
            //   %95 sağlıklı → her zaman "sağlıklı" de → %95 accuracy ama işe yaramaz!
            //   AUC tüm eşik değerlerini kapsar → daha güvenilir.
            System.out.println("\n============ 3. ROC/AUC ============\n");

            var rfF = RandomForest.fit(formula, trDf,
                    new RandomForest.Options(100, 0, SplitRule.GINI, 20, 0, 5, 1.0, null, null, null));
            var gbF = GradientTreeBoost.fit(formula, trDf,
                    new GradientTreeBoost.Options(100, 6, 32, 5, 0.1, 0.8, null, null));
            var lrF = LogisticRegression.fit(XTr, yTr);

            double[] auc = new double[3];
            for (int mi = 0; mi < 3; mi++) {
                double[] probs = new double[XTe.length];
                for (int i = 0; i < XTe.length; i++) {
                    double[] p = new double[2];
                    if (mi == 0) { rfF.predict(teDf.get(i), p); }
                    else if (mi == 1) { gbF.predict(teDf.get(i), p); }
                    else { lrF.predict(XTe[i], p); }
                    probs[i] = p[1];
                }
                // AUC inline (trapez)
                Integer[] si = new Integer[XTe.length];
                for (int i = 0; i < XTe.length; i++) si[i] = i;
                Arrays.sort(si, (a, b) -> Double.compare(probs[b], probs[a]));
                int tP = 0, tN = 0; for (int g : yTe) { if (g == 1) tP++; else tN++; }
                double a = 0, oTpr = 0, oFpr = 0; int tpS = 0, fpS = 0;
                for (int i = 0; i < XTe.length; i++) {
                    if (yTe[si[i]] == 1) tpS++; else fpS++;
                    double tpr = (double) tpS / tP, fpr = (double) fpS / tN;
                    a += (fpr - oFpr) * (tpr + oTpr) / 2.0; oTpr = tpr; oFpr = fpr;
                }
                auc[mi] = a;
                System.out.printf("%-22s: AUC = %.4f\n", mIsim[mi], auc[mi]);
            }

            // ============ 4. CONFUSION MATRIX ============
            // TP: Hasta → Hasta (doğru)    TN: Sağlıklı → Sağlıklı (doğru)
            // FP: Sağlıklı → Hasta (alarm) FN: Hasta → Sağlıklı (TEHLİKELİ!)
            //
            // Sensitivity (Recall) = TP/(TP+FN) → hastaları yakalama oranı
            // Specificity = TN/(TN+FP) → sağlıklıları doğru ayırt etme
            // Tıpta FN en tehlikeli → Sensitivity öncelikli olmalı
            System.out.println("\n============ 4. CONFUSION MATRIX ============\n");

            int tp = 0, tn = 0, fp = 0, fn = 0;
            for (int i = 0; i < XTe.length; i++) {
                int t = rfF.predict(teDf.get(i));
                if (yTe[i] == 1 && t == 1) tp++; else if (yTe[i] == 0 && t == 0) tn++;
                else if (yTe[i] == 0 && t == 1) fp++; else fn++;
            }
            double sens = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0;
            double spec = (tn + fp) > 0 ? (double) tn / (tn + fp) : 0;
            System.out.printf("RF: TP=%d TN=%d FP=%d FN=%d\n", tp, tn, fp, fn);
            System.out.printf("Sensitivity=%.4f  Specificity=%.4f\n", sens, spec);

            // ============ 5. GRID SEARCH ============
            // Hiperparametre kombinasyonlarını sistematik dener.
            // Her kombinasyon CV ile değerlendirilir → en yüksek CV Acc → optimal.
            // ntrees: Ağaç sayısı (fazla → yavaş ama stabil)
            // maxDepth: Ağaç derinliği (derin → overfitting riski)
            System.out.println("\n============ 5. GRID SEARCH ============\n");

            int[] ntArr = {50, 100, 200}; int[] mdArr = {5, 10, 20};
            double best = 0; int bestNt = 100, bestMd = 20;
            for (int nt : ntArr) for (int md : mdArr) {
                Bag[] gF = CrossValidation.stratify(y, 5); double tA = 0;
                for (int f = 0; f < 5; f++) {
                    DataFrame fTr = tumDf.get(Index.of(gF[f].samples())), fTe = tumDf.get(Index.of(gF[f].oob()));
                    var mod = RandomForest.fit(formula, fTr,
                            new RandomForest.Options(nt, 0, SplitRule.GINI, md, 0, 5, 1.0, null, null, null));
                    int[] ge = formula.y(fTe).toIntArray(); int d = 0;
                    for (int i = 0; i < fTe.nrow(); i++) if (mod.predict(fTe.get(i)) == ge[i]) d++;
                    tA += (double) d / fTe.nrow();
                }
                double cv = tA / 5.0;
                System.out.printf("nt=%d md=%d → %.4f%s\n", nt, md, cv, cv > best ? " ← best" : "");
                if (cv > best) { best = cv; bestNt = nt; bestMd = md; }
            }

            // ============ 6. OVERFITTING TESTİ ============
            // Train Acc >> Test Acc → overfitting (ezberlemiş)
            // Her ikisi düşük → underfitting (yetersiz)
            // Fark < %10 → iyi uyum
            System.out.println("\n============ 6. OVERFITTING TESTİ ============\n");

            var optRf = RandomForest.fit(formula, trDf,
                    new RandomForest.Options(bestNt, 0, SplitRule.GINI, bestMd, 0, 5, 1.0, null, null, null));
            int trD = 0; int[] trG = formula.y(trDf).toIntArray();
            for (int i = 0; i < trDf.nrow(); i++) if (optRf.predict(trDf.get(i)) == trG[i]) trD++;
            int teD = 0; int[] teG = formula.y(teDf).toIntArray();
            for (int i = 0; i < teDf.nrow(); i++) if (optRf.predict(teDf.get(i)) == teG[i]) teD++;
            double trA = (double) trD / trDf.nrow(), teA = (double) teD / teDf.nrow();
            System.out.printf("Train=%.4f Test=%.4f Fark=%.4f → %s\n", trA, teA, trA - teA,
                    trA - teA > 0.10 ? "OVERFITTING" : trA < 0.70 ? "UNDERFITTING" : "İYİ UYUM");

            // ============ 7. FİNAL + DEMO ============
            // CV Accuracy (%50 ağırlık) + AUC (%50 ağırlık) → final model
            System.out.println("\n============ 7. FİNAL + DEMO ============\n");

            double[] skor = new double[3];
            for (int i = 0; i < 3; i++) skor[i] = cvOrt[i] * 0.5 + auc[i] * 0.5;
            int bestM = 0; for (int i = 1; i < 3; i++) if (skor[i] > skor[bestM]) bestM = i;
            System.out.printf("Final: %s (CV=%.4f, AUC=%.4f)\n\n", mIsim[bestM], cvOrt[bestM], auc[bestM]);

            // Yeni hasta demo (normalize edilmiş)
            double[] minD = {28, 0, 0, 0, 0, 0, 80, 100, 0, 0, 0, 0, 60, 0, -2.6, 0};
            double[] maxD = {77, 1, 1, 1, 1, 1, 200, 603, 1, 1, 1, 1, 202, 1, 6.2, 2};

            // Hasta 1: Yüksek risk (55/E/ASY/Angina+)
            double[] h1 = norm(new double[]{55, 1, 0, 0, 0, 1, 140, 260, 0, 1, 0, 0, 130, 1, 2.0, 1}, minD, maxD);
            int t1; double[] p1 = new double[2];
            if (bestM == 1) { t1 = gbF.predict(df(new double[][]{h1}, new int[]{0}).get(0)); gbF.predict(df(new double[][]{h1}, new int[]{0}).get(0), p1); }
            else if (bestM == 2) { t1 = lrF.predict(h1); lrF.predict(h1, p1); }
            else { t1 = rfF.predict(df(new double[][]{h1}, new int[]{0}).get(0)); rfF.predict(df(new double[][]{h1}, new int[]{0}).get(0), p1); }
            System.out.printf("Hasta 1 (55/E/ASY): %s (P=%%%.0f)\n", t1 == 1 ? "RİSK" : "SAĞLIKLI", p1[1] * 100);

            // Hasta 2: Düşük risk (35/K/NAP/Angina-)
            double[] h2 = norm(new double[]{35, 0, 0, 0, 1, 0, 120, 200, 0, 1, 0, 0, 170, 0, 0.0, 2}, minD, maxD);
            int t2; double[] p2 = new double[2];
            if (bestM == 1) { t2 = gbF.predict(df(new double[][]{h2}, new int[]{0}).get(0)); gbF.predict(df(new double[][]{h2}, new int[]{0}).get(0), p2); }
            else if (bestM == 2) { t2 = lrF.predict(h2); lrF.predict(h2, p2); }
            else { t2 = rfF.predict(df(new double[][]{h2}, new int[]{0}).get(0)); rfF.predict(df(new double[][]{h2}, new int[]{0}).get(0), p2); }
            System.out.printf("Hasta 2 (35/K/NAP): %s (P=%%%.0f)\n", t2 == 1 ? "RİSK" : "SAĞLIKLI", p2[1] * 100);

            System.out.println("\n=== PROJE 2 TAMAMLANDI ===");

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /** Tekil örneği bilinen min/max ile normalize et. */
    private static double[] norm(double[] h, double[] mn, double[] mx) {
        double[] r = new double[h.length];
        for (int j = 0; j < h.length; j++) r[j] = Math.max(0, Math.min(1, (h[j] - mn[j]) / (mx[j] - mn[j] + 1e-10)));
        return r;
    }

    /** DataFrame oluştur. */
    private static DataFrame df(double[][] X, int[] y) {
        int n = X.length, m = X[0].length;
        double[][] v = new double[n][m + 1];
        for (int i = 0; i < n; i++) { System.arraycopy(X[i], 0, v[i], 0, m); v[i][m] = y[i]; }
        String[] nm = new String[m + 1]; System.arraycopy(FN, 0, nm, 0, m); nm[m] = "Hedef";
        return DataFrame.of(v, nm).factorize("Hedef");
    }
}
