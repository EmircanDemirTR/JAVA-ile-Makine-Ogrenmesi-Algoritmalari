package com.btkakademi.ml.bolum11;

import weka.clusterers.HierarchicalClusterer;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import java.io.InputStream;

public class WekaHiyerarsikKumeleme {
    static void main() {
        try {
            InputStream is = WekaHiyerarsikKumeleme.class.getClassLoader().getResourceAsStream("datasets/wholesale_customers.csv");
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(is);

            Instances veri = csvLoader.getDataSet();

            // Remove ile kategorik ve kümeleme için uygun olmayan sütunların kaldırılması
            Remove remove = new Remove();
            remove.setAttributeIndices("1,2");
            remove.setInputFormat(veri);
            Instances harcamaVeri = Filter.useFilter(veri, remove);

            // Normalizasyon
            Normalize normalize = new Normalize();
            normalize.setInputFormat(harcamaVeri);
            Instances normVeri = Filter.useFilter(harcamaVeri, normalize);

            // Kümeleme - Ward Linkage
            HierarchicalClusterer wardHC = new HierarchicalClusterer();
            wardHC.setNumClusters(3);
            wardHC.setLinkType(new SelectedTag(5, HierarchicalClusterer.TAGS_LINK_TYPE));
            wardHC.buildClusterer(normVeri);

            int[] wardAtamalar = kumeAtamalariniAl(wardHC, normVeri);
            int[] wardBoyutlar = kumeBoyutHesapla(wardAtamalar, 3);

            for (int i = 0; i < 3; i++) {
                System.out.printf("Kume %d: %d müşteri\n", i, wardBoyutlar[i]);
            }

            kumeProfilYazdir(harcamaVeri, wardAtamalar, 3);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }


    private static int[] kumeAtamalariniAl(HierarchicalClusterer hc, Instances veri) throws Exception {
        int[] atamalar = new int[veri.numInstances()];
        for (int i = 0; i < veri.numInstances(); i++) {
            atamalar[i] = hc.clusterInstance(veri.instance(i));
        }
        return atamalar;
    }

    private static int[] kumeBoyutHesapla(int[] atamalar, int k) {
        int[] boyutlar = new int[k];
        for (int atama : atamalar) {
            boyutlar[atama]++;
        }
        return boyutlar;
    }

    private static void kumeProfilYazdir(Instances veri, int[] atamalar, int k) {
        System.out.println("Kume Harcamalara Profilleri");
        System.out.printf("  %-8s", "Küme");

        for (int j = 0; j < veri.numAttributes(); j++) {
            System.out.printf(" %12s", veri.attribute(j).name().length() > 12 ?
                    veri.attribute(j).name().substring(0, 12) : veri.attribute(j).name());
        }
        System.out.println();

        for (int c = 0; c < k; c++) {
            double[] toplamlar = new double[veri.numAttributes()];
            int sayac = 0;
            for (int i = 0; i < veri.numInstances(); i++) {
                if (atamalar[i] == c) {
                    for (int j = 0; j < veri.numAttributes(); j++) {
                        toplamlar[j] += veri.instance(i).value(j);
                    }
                    sayac++;
                }
            }
            System.out.printf("  Küme: %d", c);
            for (int j = 0; j < veri.numAttributes(); j++) {
                System.out.printf(" %12.0f", toplamlar[j] / sayac);
            }
            System.out.println();
        }
    }
}
