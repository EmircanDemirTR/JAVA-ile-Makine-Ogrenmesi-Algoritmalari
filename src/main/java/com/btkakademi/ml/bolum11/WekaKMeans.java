package com.btkakademi.ml.bolum11;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.InputStream;

public class WekaKMeans {
    static void main() {
        try {
            InputStream is = WekaKMeans.class.getClassLoader().getResourceAsStream("datasets/iris.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            int[] gercekEtiketler = new int[veri.numInstances()];
            for (int i = 0; i < veri.numInstances(); i++) {
                gercekEtiketler[i] = (int) veri.instance(i).classValue();
            }

            // Remove filtresi ile Class'ı veriden çıkartalım
            Remove remove = new Remove();
            remove.setAttributeIndices("last");
            remove.setInputFormat(veri);

            Instances kumelemeVeri = Filter.useFilter(veri, remove);

            // Simple K-Means Kümeleme
            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(3);
            kmeans.setSeed(42);
            kmeans.setPreserveInstancesOrder(true);

            long baslangic = System.currentTimeMillis();
            kmeans.buildClusterer(kumelemeVeri);
            long sure = System.currentTimeMillis() - baslangic;
            System.out.println("K-Means modeli eğitildi ve süresi: " + sure + "ms");

            int[] kmeansAtamalar = kmeans.getAssignments();

            // Her örneğin küme indeksi
            double[] kmeansBoyutlar = kmeans.getClusterSizes();
            System.out.println("Küme boyutları");
            for (int i = 0; i < 3; i++) {
                System.out.println("Küme: " + i + " için: " + kmeansBoyutlar[i] + " ornek");
            }

            // Küme Merkezleri
            Instances merkezler = kmeans.getClusterCentroids();
            System.out.println("Küme Merkezleri: ");
            System.out.printf("  %-8s %-12s %-12s %-12s %-12s\n", "Küme", "SepalLen", "SepalWid", "PetalLen", "PetalWid");
            for (int i = 0; i < merkezler.numInstances(); i++) {
                System.out.printf(" Küme %d: %12.4f %12.4f %12.4f %12.4f\n", i,
                        merkezler.instance(i).value(0), merkezler.instance(i).value(1),
                        merkezler.instance(i).value(2), merkezler.instance(i).value(3)
                );
            }
            System.out.println("SSE: " + kmeans.getSquaredError());

            // Kümeleme Değerlendirme
            ClusterEvaluation kmeansEval = new ClusterEvaluation();
            kmeansEval.setClusterer(kmeans);
            kmeansEval.evaluateClusterer(kumelemeVeri);
            System.out.println("K-Means Kümeleme");
            System.out.println(kmeansEval.clusterResultsToString());


            // Soft Clustering - EM
            EM em = new EM();
            em.setNumClusters(3);
            em.setSeed(42);

            em.buildClusterer(kumelemeVeri);

            ClusterEvaluation emEval = new ClusterEvaluation();
            emEval.setClusterer(em);
            emEval.evaluateClusterer(kumelemeVeri);
            System.out.println(emEval.clusterResultsToString());

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
