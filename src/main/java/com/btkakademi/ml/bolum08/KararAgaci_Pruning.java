package com.btkakademi.ml.bolum08;

// Weka kütüphane import'ları
import weka.classifiers.Evaluation;        // Model değerlendirme sınıfı - doğruluk, kappa vb.
import weka.classifiers.trees.J48;          // J48 karar ağacı - C4.5 algoritmasının implementasyonu
import weka.core.Instances;                // Weka veri yapısı - satır ve sütunları tutar
import weka.core.converters.ArffLoader;   // ARFF dosya okuyucu

import java.util.Random;  // Rastgele sayı üreteci - veri karıştırma için

/**
 * Karar Ağacı Pruning (Budama) Optimizasyonu
 * BTK Akademi - Java ile Makine Öğrenmesi
 * Video: 8.11
 *
 * ======================================================
 * VİDEO GİRİŞ METNİ
 * ======================================================
 * "Bu derste Glass veri seti üzerinde pruning optimizasyonu
 * yapacağız. Glass veri seti, camların kimyasal bileşimine
 * göre 6 farklı türe sınıflandırıyor. 214 örnek ve 9 özellik
 * var. Daha fazla sınıf olduğu için pruning etkisi daha
 * belirgin olacak. Unpruned vs pruned karşılaştırması yapıp,
 * grid search ile optimal parametreleri bulacağız."
 * ======================================================
 *
 * GLASS VERİ SETİ:
 * - 214 örnek, 9 özellik, 6 sınıf (cam türü: 1,2,3,5,6,7)
 * - Özellikler: RI (kırılma indisi), Na, Mg, Al, Si, K, Ca, Ba, Fe
 * - Adli tıpta cam parçası analizi için kullanılır
 */
public class KararAgaci_Pruning {

    public static void main(String[] args) {
        try {
            // Program başlığını yazdır
            System.out.println("=== KARAR AGACI PRUNING OPTIMIZASYONU ===\n");

            // ============================================================
            // 1. VERİ YÜKLEME
            // ============================================================
            // Classpath'ten ARFF dosyasını oku
            var is = KararAgaci_Pruning.class.getClassLoader().getResourceAsStream("datasets/glass.arff");
            // Dosya bulunamazsa hata fırlat
            if (is == null) throw new RuntimeException("glass.arff bulunamadi!");

            // ArffLoader ile veri yükle
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);
            Instances veri = loader.getDataSet();
            // Son sütunu hedef değişken olarak ayarla
            veri.setClassIndex(veri.numAttributes() - 1);

            System.out.println("Veri yuklendi: " + veri.numInstances() + " ornek");

            // ============================================================
            // 2. TRAIN/TEST SPLIT
            // ============================================================
            // Veriyi rastgele karıştır (seed=42 tekrarlanabilirlik için)
            veri.randomize(new Random(42));
            // %70 eğitim hesapla
            int egitimBoyut = (int) (veri.numInstances() * 0.7);
            // Eğitim ve test setlerini oluştur
            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Egitim: " + egitim.numInstances() + ", Test: " + test.numInstances());

            // ============================================================
            // 3. PRUNED vs UNPRUNED KARŞILAŞTIRMASI
            // ============================================================
            // Pruning (Budama): Ağacın gereksiz dallarını keserek overfitting'i önler
            //
            // NEDEN BUDAMA YAPARIZ?
            // - Unpruned ağaç: Eğitim verisine çok iyi uyar ama test verisinde kötü
            // - Pruned ağaç: Daha genel, yeni veriye daha iyi adapte olur
            //
            // PRUNING TİPLERİ:
            // 1. Pre-pruning: Ağaç büyürken durdur (maxDepth, minLeafSize)
            // 2. Post-pruning: Ağaç oluştuktan sonra buda (J48 varsayılan)
            //
            // J48 POST-PRUNING NASIL ÇALIŞIR?
            // - Subtree replacement: Alt ağacı yaprak düğümle değiştir
            // - Confidence-based: İstatistiksel test ile dalı kes
            System.out.println("\n--- PRUNED vs UNPRUNED ---\n");

            // UNPRUNED (budanmamış) ağaç
            J48 unpruned = new J48();
            // setUnpruned(true): Budama yapma, tam ağacı oluştur
            unpruned.setUnpruned(true);
            unpruned.buildClassifier(egitim);

            // Budanmamış modeli test et
            Evaluation evalUnpruned = new Evaluation(egitim);
            evalUnpruned.evaluateModel(unpruned, test);
            int sizeUnpruned = (int) unpruned.measureTreeSize();
            int leavesUnpruned = (int) unpruned.measureNumLeaves();

            System.out.printf("Unpruned: Dogruluk=%.2f%%, Agac=%d dugum, %d yaprak%n",
                    evalUnpruned.pctCorrect(), sizeUnpruned, leavesUnpruned);

            // PRUNED (budanmış) ağaç - varsayılan ayarlar
            J48 pruned = new J48();
            pruned.buildClassifier(egitim);

            Evaluation evalPruned = new Evaluation(egitim);
            evalPruned.evaluateModel(pruned, test);
            int sizePruned = (int) pruned.measureTreeSize();
            int leavesPruned = (int) pruned.measureNumLeaves();

            System.out.printf("Pruned:   Dogruluk=%.2f%%, Agac=%d dugum, %d yaprak%n",
                    evalPruned.pctCorrect(), sizePruned, leavesPruned);

            // Küçülme oranını hesapla
            int kuculme = (int) Math.round((1 - (double) sizePruned / sizeUnpruned) * 100);
            System.out.println("\nBudama ile agac %" + kuculme + " kucultuldu");

            // ============================================================
            // 4. CONFIDENCE FACTOR GRID SEARCH
            // ============================================================
            // Confidence Factor (CF): Budama agresifliğini kontrol eder
            //
            // NASIL ÇALIŞIR?
            // - J48 her düğümde "Bu dalı kesersem hata artar mı?" sorar
            // - CF, bu kararın güven seviyesini belirler
            // - Düşük CF (0.05): Çok agresif budama, çok basit ağaç
            // - Yüksek CF (0.5): Az budama, karmaşık ağaç
            // - Varsayılan: 0.25
            //
            // TRADE-OFF:
            // - Çok düşük CF → Underfitting riski (çok basit model)
            // - Çok yüksek CF → Overfitting riski (ezberleme)
            System.out.println("\n--- CONFIDENCE FACTOR SECIMI ---\n");

            // Denenecek CF değerleri
            float[] cfDegerleri = {0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f};
            double enIyiDogruluk = 0;
            float enIyiCF = 0.25f;

            // Her CF değerini dene
            for (float cf : cfDegerleri) {
                J48 tree = new J48();
                // setConfidenceFactor: Pruning için güven eşiği
                tree.setConfidenceFactor(cf);
                tree.buildClassifier(egitim);

                Evaluation eval = new Evaluation(egitim);
                eval.evaluateModel(tree, test);

                int size = (int) tree.measureTreeSize();
                double acc = eval.pctCorrect();

                // En iyi sonuç mu kontrol et
                String marker = "";
                if (acc > enIyiDogruluk) {
                    enIyiDogruluk = acc;
                    enIyiCF = cf;
                    marker = " <-- EN IYI";
                }
                System.out.printf("CF=%.2f -> Dogruluk=%.2f%%, Boyut=%d%s%n", cf, acc, size, marker);
            }

            System.out.println("\nSecilen CF: " + enIyiCF);

            // ============================================================
            // 5. MIN NUM OBJ GRID SEARCH
            // ============================================================
            // minNumObj: Yaprak düğümde minimum örnek sayısı
            //
            // NASIL ÇALIŞIR?
            // - Bir yaprak en az bu kadar örnek içermeli
            // - Düşük değer (1-2): Tek örneğe bile dal oluşur → Overfitting
            // - Yüksek değer (20+): Çok genel kurallar → Underfitting
            // - Varsayılan: 2
            //
            // PRATİK KURAL:
            // - Küçük veri setleri (< 1000): minObj = 2-5
            // - Orta veri setleri (1000-10000): minObj = 5-10
            // - Büyük veri setleri (> 10000): minObj = 10-50
            System.out.println("\n--- MIN NUM OBJ SECIMI ---\n");

            int[] minObjDegerleri = {1, 2, 3, 5, 10, 15, 20};
            enIyiDogruluk = 0;
            int enIyiMinObj = 2;

            // Her minNumObj değerini dene
            for (int minObj : minObjDegerleri) {
                J48 tree = new J48();
                tree.setConfidenceFactor(enIyiCF);
                // setMinNumObj: Yapraktaki minimum örnek sayısı
                tree.setMinNumObj(minObj);
                tree.buildClassifier(egitim);

                Evaluation eval = new Evaluation(egitim);
                eval.evaluateModel(tree, test);

                int leaves = (int) tree.measureNumLeaves();
                double acc = eval.pctCorrect();

                String marker = "";
                if (acc > enIyiDogruluk) {
                    enIyiDogruluk = acc;
                    enIyiMinObj = minObj;
                    marker = " <-- EN IYI";
                }
                System.out.printf("minObj=%2d -> Dogruluk=%.2f%%, Yaprak=%d%s%n", minObj, acc, leaves, marker);
            }

            System.out.println("\nSecilen minObj: " + enIyiMinObj);

            // ============================================================
            // 6. FINAL MODEL
            // ============================================================
            System.out.println("\n--- FINAL MODEL ---");

            // En iyi parametrelerle final model oluştur
            J48 finalTree = new J48();
            finalTree.setConfidenceFactor(enIyiCF);
            finalTree.setMinNumObj(enIyiMinObj);
            finalTree.buildClassifier(egitim);

            Evaluation finalEval = new Evaluation(egitim);
            finalEval.evaluateModel(finalTree, test);

            // Final sonuçları yazdır
            System.out.printf("CF=%.2f, minObj=%d%n", enIyiCF, enIyiMinObj);
            System.out.printf("Dogruluk: %.2f%%, Kappa: %.4f%n", finalEval.pctCorrect(), finalEval.kappa());
            System.out.printf("Agac: %d dugum, %d yaprak%n",
                    (int) finalTree.measureTreeSize(), (int) finalTree.measureNumLeaves());

            // Ağaç yapısını göster
            System.out.println("\nKarar Agaci:");
            System.out.println(finalTree);

            System.out.println("=== PRUNING OPTIMIZASYONU TAMAMLANDI ===");

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
