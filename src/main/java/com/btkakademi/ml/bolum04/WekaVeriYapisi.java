package com.btkakademi.ml.bolum04;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * WEKA Veri Yapisi Ornegi
 * Bu sinif WEKA kutuphanesindeki temel veri yapilarini gosterir.
 * 
 * WEKA'da temel yapilar:
 * - Instances: Tum veri setini temsil eder (tablo)
 * - Instance: Tek bir veri satirini temsil eder (satir)
 * - Attribute: Bir ozelligi/sutunu temsil eder (sutun)
 */
public class WekaVeriYapisi {

    public static void main(String[] args) {

        // ============================================================
        // ADIM 1: ATTRIBUTE (OZELLIK/SUTUN) TANIMLAMA
        // ============================================================

        // Sayisal ozellikler (Numeric Attribute)
        // Surekli degerler alabilir: 25, 35.5, 100 gibi
        Attribute yasAttribute = new Attribute("yas");
        Attribute maasAttribute = new Attribute("maas");

        // Kategorik ozellik - Cinsiyet (Nominal Attribute)
        // Sadece belirlenen degerlerden birini alabilir
        // Diamond operator (<>) ile modern Java kullanimi
        ArrayList<String> cinsiyetDegerleri = new ArrayList<>();
        cinsiyetDegerleri.add("Erkek");          // index 0
        cinsiyetDegerleri.add("Kadin");          // index 1
        Attribute cinsiyetAttribute = new Attribute("cinsiyet", cinsiyetDegerleri);

        // Kategorik ozellik - Satin alma durumu (hedef degisken)
        ArrayList<String> satinAlmaDegerleri = new ArrayList<>();
        satinAlmaDegerleri.add("Hayir");         // index 0
        satinAlmaDegerleri.add("Evet");          // index 1
        Attribute satinAlmaAttribute = new Attribute("satin_alma", satinAlmaDegerleri);

        // ============================================================
        // ADIM 2: INSTANCES (VERI SETI) OLUSTURMA
        // ============================================================

        // Tum ozellikleri bir listeye ekle
        ArrayList<Attribute> ozellikListesi = new ArrayList<>();
        ozellikListesi.add(yasAttribute);        // index 0
        ozellikListesi.add(maasAttribute);       // index 1
        ozellikListesi.add(cinsiyetAttribute);   // index 2
        ozellikListesi.add(satinAlmaAttribute);  // index 3

        // Instances olustur (veri seti)
        // Parametreler: isim, ozellik listesi, baslangic kapasitesi
        Instances veriSeti = new Instances("MusteriVerisi", ozellikListesi, 10);

        // Sinif (hedef) ozelligini belirle - son sutun
        veriSeti.setClassIndex(veriSeti.numAttributes() - 1);

        // ============================================================
        // ADIM 3: INSTANCE (VERI SATIRLARI) EKLEME
        // ============================================================

        // 1. Musteri: 25 yas, 5000 TL maas, Erkek, Satin Aldi
        Instance musteri1 = new DenseInstance(4);
        musteri1.setValue(yasAttribute, 25);
        musteri1.setValue(maasAttribute, 5000);
        musteri1.setValue(cinsiyetAttribute, "Erkek");
        musteri1.setValue(satinAlmaAttribute, "Evet");
        veriSeti.add(musteri1);

        // 2. Musteri: 35 yas, 8000 TL maas, Kadin, Satin Aldi
        Instance musteri2 = new DenseInstance(4);
        musteri2.setValue(yasAttribute, 35);
        musteri2.setValue(maasAttribute, 8000);
        musteri2.setValue(cinsiyetAttribute, "Kadin");
        musteri2.setValue(satinAlmaAttribute, "Evet");
        veriSeti.add(musteri2);

        // 3. Musteri: 22 yas, 3000 TL maas, Erkek, Satin Almadi
        Instance musteri3 = new DenseInstance(4);
        musteri3.setValue(yasAttribute, 22);
        musteri3.setValue(maasAttribute, 3000);
        musteri3.setValue(cinsiyetAttribute, "Erkek");
        musteri3.setValue(satinAlmaAttribute, "Hayir");
        veriSeti.add(musteri3);

        // 4. Musteri: 45 yas, 12000 TL maas, Kadin, Satin Aldi
        Instance musteri4 = new DenseInstance(4);
        musteri4.setValue(yasAttribute, 45);
        musteri4.setValue(maasAttribute, 12000);
        musteri4.setValue(cinsiyetAttribute, "Kadin");
        musteri4.setValue(satinAlmaAttribute, "Evet");
        veriSeti.add(musteri4);

        // 5. Musteri: 30 yas, 4500 TL maas, Erkek, Satin Almadi
        Instance musteri5 = new DenseInstance(4);
        musteri5.setValue(yasAttribute, 30);
        musteri5.setValue(maasAttribute, 4500);
        musteri5.setValue(cinsiyetAttribute, "Erkek");
        musteri5.setValue(satinAlmaAttribute, "Hayir");
        veriSeti.add(musteri5);

        // ============================================================
        // ADIM 4: VERI SETINI INCELEME
        // ============================================================

        System.out.println("========================================");
        System.out.println("WEKA VERI YAPISI ORNEGI");
        System.out.println("========================================\n");

        // Veri seti bilgileri
        System.out.println("--- Veri Seti Bilgileri ---");
        System.out.println("Veri seti adi: " + veriSeti.relationName());
        System.out.println("Toplam ornek sayisi: " + veriSeti.numInstances());
        System.out.println("Toplam ozellik sayisi: " + veriSeti.numAttributes());
        System.out.println("Sinif ozelligi: " + veriSeti.classAttribute().name());
        System.out.println();

        // Ozellikleri listele
        System.out.println("--- Ozellikler (Attributes) ---");
        for (int i = 0; i < veriSeti.numAttributes(); i++) {
            Attribute attr = veriSeti.attribute(i);
            String tip = attr.isNumeric() ? "Sayisal" : "Kategorik";
            System.out.println(i + ". " + attr.name() + " (" + tip + ")");
        }
        System.out.println();

        // Tum veriyi yazdir
        System.out.println("--- Veri Seti Icerigi (ARFF Formati) ---");
        System.out.println(veriSeti);

        // ============================================================
        // ADIM 5: TEK BIR INSTANCE'A ERISME
        // ============================================================

        System.out.println("--- Tek Bir Ornege Erisim ---");
        Instance ilkMusteri = veriSeti.instance(0);
        System.out.println("Ilk musteri (index 0): " + ilkMusteri);
        System.out.println("  - Yas: " + ilkMusteri.value(yasAttribute));
        System.out.println("  - Maas: " + ilkMusteri.value(maasAttribute));
        System.out.println("  - Cinsiyet: " + ilkMusteri.stringValue(cinsiyetAttribute));
        System.out.println("  - Satin Alma: " + ilkMusteri.stringValue(satinAlmaAttribute));
    }
}