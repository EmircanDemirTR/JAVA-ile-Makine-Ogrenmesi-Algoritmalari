package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class NaiveBayes_ModelKarsilastirma {
    static void main() {
        try {
            var is = NaiveBayes_ModelKarsilastirma.class.getClassLoader().getResourceAsStream("datasets/glass.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Train Test Split
            veri.randomize(new Random(42));

            int egitimBoyut = (int) (veri.numInstances() * 0.7);

            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            
            // Model Karşılaştırma
            double enIyiDogruluk = 0;
            String enIyiModel = "";


            // Gaussian Naive Bayes
            NaiveBayes gaussianNB = new NaiveBayes();
            gaussianNB.buildClassifier(egitim);

            Evaluation evalGaussian = new Evaluation(egitim);
            evalGaussian.evaluateModel(gaussianNB, test);

            double accGaussian = evalGaussian.pctCorrect();
            System.out.println("Dogruluk: " + accGaussian);
            System.out.println("Kappa: " + evalGaussian.kappa());

            if (accGaussian > enIyiDogruluk) {
                enIyiDogruluk = accGaussian;
                enIyiModel = "Gaussian NB";
            }


            // Kernel Density
            NaiveBayes kernelNB = new NaiveBayes();
            kernelNB.setUseKernelEstimator(true);
            kernelNB.buildClassifier(egitim);

            Evaluation evalKernel = new Evaluation(egitim);
            evalKernel.evaluateModel(kernelNB, test);

            double accKernel = evalKernel.pctCorrect();
            System.out.println("Dogruluk: " + accKernel);
            System.out.println("Kappa: " + evalKernel.kappa());

            if (accGaussian > enIyiDogruluk) {
                enIyiDogruluk = accKernel;
                enIyiModel = "Kernel Density";
            }


            // Supervised Discretization
            NaiveBayes discNB = new NaiveBayes();
            discNB.setUseSupervisedDiscretization(true);
            discNB.buildClassifier(egitim);

            Evaluation evalDisc = new Evaluation(egitim);
            evalDisc.evaluateModel(discNB, test);

            double accDisc = evalDisc.pctCorrect();
            System.out.println("Dogruluk: " + accDisc);
            System.out.println("Kappa: " + evalDisc.kappa());

            if (accDisc > enIyiDogruluk) {
                enIyiDogruluk = accDisc;
                enIyiModel = "Supervised Discretization";
            }

            System.out.println("En iyi model: " + enIyiModel);
            System.out.println("En iyi dogruluk: " + enIyiDogruluk);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
