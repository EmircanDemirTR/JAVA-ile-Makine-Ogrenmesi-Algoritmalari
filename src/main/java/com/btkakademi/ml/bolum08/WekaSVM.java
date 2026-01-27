package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.Random;

public class WekaSVM {
    static void main() {
        try {
            var is = WekaSVM.class.getClassLoader().getResourceAsStream("datasets/wine.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Normalizasyon
            Normalize normalize = new Normalize();
            normalize.setInputFormat(veri);
            veri = Filter.useFilter(veri, normalize);

            // Train Test Split

            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);

            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Egitim: " + egitim.numInstances() + ", Test: " + test.numInstances());

            // Model Egitimi - SMO (RBF Kernel)

            SMO svm = new SMO();
            svm.setC(1.0);

            // RBF Kernel
            RBFKernel kernel = new RBFKernel();
            kernel.setGamma(1.0);
            svm.setKernel(kernel);

            svm.buildClassifier(egitim);

            // Değerlendirme
            Evaluation eval = new Evaluation(egitim);
            eval.evaluateModel(svm, test);

            System.out.println("Dogruluk Degeri: " + eval.pctCorrect());
            System.out.println("Kappa: " + eval.kappa());

            // Karmaşıklık Matrisi
            System.out.println(eval.toMatrixString());

            // C Parametresi Grid Search

            double[] cDegerleri = {0.01, 0.1, 1.0, 10.0, 100.0};
            double enIyiDogruluk = 0;
            double enIyiC = 1.0;

            for (double c : cDegerleri) {
                SMO svmGrid = new SMO();
                svmGrid.setC(c);
                RBFKernel kernelGrid = new RBFKernel();
                kernelGrid.setGamma(1.0);
                svmGrid.setKernel(kernelGrid);

                svmGrid.buildClassifier(egitim);

                Evaluation evalGrid = new Evaluation(egitim);
                evalGrid.evaluateModel(svmGrid, test);
                double acc = evalGrid.pctCorrect();

                String marker = "";
                if (acc > enIyiDogruluk) {
                    enIyiDogruluk = acc;
                    enIyiC = c;
                    marker = "<---- EN IYI";
                }
                System.out.printf("C=%.2f -> %.2f%%%s%n", c, acc, marker);
            }
            System.out.println("Secilen C: " + enIyiC);


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
