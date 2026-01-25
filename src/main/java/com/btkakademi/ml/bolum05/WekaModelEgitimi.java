package com.btkakademi.ml.bolum05;

import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class WekaModelEgitimi {
    public static void main(String[] args) throws Exception {

        // ARFF dosyasini yukle
        var is = WekaModelEgitimi.class.getClassLoader().getResourceAsStream("datasets/iris.arff");
        ArffLoader loader = new ArffLoader();
        loader.setSource(is);

        Instances veri = loader.getDataSet();
        veri.setClassIndex(veri.numAttributes() - 1);

        System.out.println("Veri seti yuklendi: " + veri.numInstances() + " ornek");

        // 2. J48 Decision Tree modeli eğit
        J48 model = new J48();
        model.buildClassifier(veri);

        // 3. Yeni Veri ile Tahmin Yapma
        double[] yeniVeri = {5.1, 3.5, 1.4, 0.2};
        DenseInstance ornek = new DenseInstance(1.0, yeniVeri);
        ornek.setDataset(veri);

        double tahmin = model.classifyInstance(ornek);
        String sinif = veri.classAttribute().value((int) tahmin);

        System.out.println("Tahminim: " + sinif);
    }
}
