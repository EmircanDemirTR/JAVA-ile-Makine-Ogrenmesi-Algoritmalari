package com.btkakademi.ml.bolum08;

// Weka kütüphane import'ları
import weka.classifiers.trees.J48;          // J48 karar ağacı - C4.5 algoritması
import weka.core.Instances;                // Weka veri yapısı - veri çerçevesi
import weka.core.converters.ArffLoader;   // ARFF dosya okuyucu

import java.util.Random;  // Rastgele sayı üreteci

/**
 * Karar Ağacı Görselleştirme ve Yorumlama
 * BTK Akademi - Java ile Makine Öğrenmesi
 * Video: 8.12
 *
 * ======================================================
 * VİDEO GİRİŞ METNİ
 * ======================================================
 * "Bu derste Wine veri seti üzerinde karar ağacı görselleştirme
 * ve yorumlama yapacağız. Ağaç yapısını metin olarak okuyup,
 * if-then kurallarına dönüştüreceğiz. Ayrıca Graphviz DOT
 * formatına export edip, görsel olarak da inceleyebileceğiz.
 * Wine veri setinde 13 kimyasal özellik var - bakalım ağaç
 * hangi özellikleri en önemli buluyor."
 * ======================================================
 *
 * WINE VERİ SETİ:
 * - 178 örnek, 13 özellik, 3 sınıf (şarap türü: 1, 2, 3)
 * - Özellikler: alcohol, malic_acid, ash, alcalinity, magnesium,
 *   phenols, flavanoids, nonflavanoid, proanthocyanins,
 *   color_intensity, hue, od280, proline
 */
public class KararAgaci_Gorsellestirme {

    public static void main(String[] args) {
        try {
            // Program başlığını yazdır
            System.out.println("=== KARAR AGACI GORSELLESTIRME ===\n");

            // ============================================================
            // 1. VERİ YÜKLEME
            // ============================================================
            // Wine veri setini ARFF formatından yükleyeceğiz.
            // ARFF: Attribute-Relation File Format - Weka'nın standart formatı

            // Classpath'ten ARFF dosyasını oku
            var is = KararAgaci_Gorsellestirme.class.getClassLoader().getResourceAsStream("datasets/wine.arff");
            // Dosya bulunamazsa erken başarısızlık
            if (is == null) throw new RuntimeException("wine.arff bulunamadi!");

            // ArffLoader ile veriyi yükle
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);
            // Instances: Weka'nın DataFrame benzeri veri yapısı
            Instances veri = loader.getDataSet();
            // Son sütunu hedef değişken (class) olarak ayarla
            veri.setClassIndex(veri.numAttributes() - 1);

            // Veri seti hakkında bilgi yazdır
            System.out.println("--- VERI SETI ---");
            System.out.println("Relation: " + veri.relationName());
            System.out.println("Ornek: " + veri.numInstances() + ", Ozellik: " + (veri.numAttributes() - 1));

            // ============================================================
            // 2. MODEL EĞİTİMİ
            // ============================================================
            // Görselleştirme için basit bir J48 modeli eğiteceğiz.
            // Tüm veriyi kullanarak daha anlaşılır bir ağaç elde edeceğiz.

            // Veriyi rastgele karıştır (tekrarlanabilirlik için seed=42)
            veri.randomize(new Random(42));

            // J48 modeli oluştur
            J48 tree = new J48();
            // confidenceFactor=0.25: Varsayılan budama agresifliği
            // Daha düşük değer = daha fazla budama = daha basit ağaç
            tree.setConfidenceFactor(0.25f);
            // Modeli tüm veri ile eğit (görselleştirme amaçlı)
            tree.buildClassifier(veri);

            // ============================================================
            // 3. AĞAÇ METRİKLERİ
            // ============================================================
            // Ağacın karmaşıklığını ölçen metrikleri hesaplayacağız.

            System.out.println("\n--- AGAC METRIKLERI ---");
            // measureTreeSize(): Toplam düğüm sayısı (iç düğüm + yaprak)
            System.out.printf("Toplam dugum: %.0f, Yaprak: %.0f%n",
                    tree.measureTreeSize(), tree.measureNumLeaves());

            // ============================================================
            // 4. AĞAÇ YAPISI (METİN)
            // ============================================================
            // J48.toString() ağacı metin formatında döndürür.
            // Bu çıktı if-then kuralları olarak okunabilir.
            //
            // AĞAÇ OKUMA REHBERİ:
            // - | karakteri: Derinlik seviyesini gösterir (her | bir seviye)
            // - <= veya >: Sayısal özellik için eşik değeri
            // - : karakteri: Yaprak düğümü (sınıf tahmini) işaretler
            // - (x/y): x = bu yapraktaki örnek sayısı, y = yanlış sınıflandırma
            //
            // ÖRNEK OKUMA:
            // flavanoids <= 1.575: 3 (48.0)
            // → "Eğer flavanoids ≤ 1.575 ise, sınıf 3 (48 örnek, 0 hata)"
            //
            // IF-THEN KURALI OLARAK:
            // IF flavanoids <= 1.575 THEN class = 3
            // ELSE IF proline > 755 THEN class = 1
            // ...

            System.out.println("\n--- KARAR AGACI YAPISI ---\n");
            // Ağacı metin olarak yazdır
            System.out.println(tree);

            // ============================================================
            // 5. KÖK ÖZELLİK (EN ÖNEMLİ)
            // ============================================================
            // Karar ağacında kök düğüm en yüksek information gain'e
            // sahip özelliktir - yani en ayırt edici özellik.
            //
            // INFORMATION GAIN NEDİR?
            // - Bir özelliğin veriyi ne kadar iyi ayırdığını ölçer
            // - Entropy azalması olarak hesaplanır
            // - IG(S, A) = Entropy(S) - Σ (|Sv|/|S|) * Entropy(Sv)
            //
            // C4.5 (J48) FARK:
            // - ID3: Information Gain kullanır
            // - C4.5: Gain Ratio kullanır (çok değerli özelliklere bias'ı önler)
            // - Gain Ratio = Information Gain / Split Information
            //
            // KÖK ÖZELLİK = En yüksek Gain Ratio'ya sahip özellik

            // J48'in metin çıktısını al
            String agacStr = tree.toString();
            // Satırlara böl
            String[] satirlar = agacStr.split("\n");

            // Ağaçtan kök düğümü bul (ilk bölme koşulunu içeren satır)
            System.out.println("--- KOK OZELLIK ---");
            for (String satir : satirlar) {
                // <=, >, = karakterlerinden birini içeren ilk satır
                if (satir.contains("<=") || satir.contains(">")) {
                    // | ile başlamayan satır kök düğümdür
                    if (!satir.trim().startsWith("|")) {
                        // Özellik adını çıkar (karşılaştırma operatöründen önceki kısım)
                        String kokOzellik = satir.split("[<>=]")[0].trim();
                        System.out.println("En onemli ozellik: " + kokOzellik);
                        break;
                    }
                }
            }

            // ============================================================
            // 6. GRAPHVIZ DOT FORMATI
            // ============================================================
            // J48, ağacı DOT formatında dışa aktarabilir.
            // DOT formatı Graphviz ile görselleştirilebilir.
            //
            // DOT FORMAT NEDİR?
            // - Graph Description Language
            // - Graphviz yazılımı ile PNG/SVG'ye dönüştürülebilir
            // - Online görselleştirme: https://dreampuf.github.io/GraphvizOnline
            //
            // KULLANIM:
            // 1. DOT çıktısını .dot dosyasına kaydet
            // 2. Terminal: dot -Tpng tree.dot -o tree.png
            // 3. Veya online tool'a yapıştır
            //
            // ALTERNATİF ARAÇLAR:
            // - Weka GUI'de "Visualize tree" seçeneği
            // - Python'da sklearn.tree.plot_tree()

            System.out.println("\n--- GRAPHVIZ DOT FORMATI ---\n");

            try {
                // J48.graph() metodu DOT formatı döndürür
                // DOT: Graph Description Language
                String dotFormat = tree.graph();

                // Çıktı çok uzun olabilir, ilk 600 karakteri göster
                if (dotFormat.length() > 600) {
                    // Kısaltılmış çıktı
                    System.out.println(dotFormat.substring(0, 600) + "\n...(devami var)");
                } else {
                    // Tam çıktı
                    System.out.println(dotFormat);
                }
            } catch (Exception e) {
                // DOT formatı alınamazsa hata mesajı
                System.out.println("DOT formati alinamadi: " + e.getMessage());
            }

            // Program tamamlandı
            System.out.println("\n=== GORSELLESTIRME TAMAMLANDI ===");

        } catch (Exception e) {
            // Hata durumunda detaylı bilgi ver
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
