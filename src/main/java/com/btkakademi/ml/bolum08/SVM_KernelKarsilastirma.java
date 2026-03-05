package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.Random;

public class SVM_KernelKarsilastirma {
    static void main() {
        try {
            var is = SVM_KernelKarsilastirma.class.getClassLoader().getResourceAsStream("datasets/glass.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);
            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Normalizasyon
            Normalize normalize = new Normalize();
            normalize.setInputFormat(veri);
            veri = Filter.useFilter(veri, normalize);

            // Train - Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);

            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Egitim: " + egitim.numInstances() + ", Test: " + test.numInstances());

            // Kernel Karşılaştırma
            double enIyiDogruluk = 0;
            String enIyiKernel = "";

            // Linear Kernel
            SMO svmLinear = new SMO();
            PolyKernel linearKernel = new PolyKernel();
            linearKernel.setExponent(1);
            svmLinear.setKernel(linearKernel);
            svmLinear.setC(1.0);
            svmLinear.buildClassifier(egitim);

            Evaluation evalLinear = new Evaluation(egitim);
            evalLinear.evaluateModel(svmLinear, test);
            double accLinear = evalLinear.pctCorrect();

            if (accLinear > enIyiDogruluk) {
                enIyiDogruluk = accLinear;
                enIyiKernel = "Linear";
            }
            System.out.println("Linear Doğruluk Oranı: " + accLinear);

            // Polynomial d=2
            SMO svmPoly2 = new SMO();
            PolyKernel poly2 = new PolyKernel();
            poly2.setExponent(2);
            svmPoly2.setKernel(poly2);
            svmPoly2.setC(1.0);
            svmPoly2.buildClassifier(egitim);

            Evaluation evalPoly2 = new Evaluation(egitim);
            evalPoly2.evaluateModel(svmPoly2, test);
            double accPoly2 = evalPoly2.pctCorrect();

            if (accPoly2 > enIyiDogruluk) {
                enIyiDogruluk = accPoly2;
                enIyiKernel = "Polynomial d=2";
            }
            System.out.println("Polynomial d=2 Doğruluk Oranı: " + accPoly2);


            // Polynomial d=3
            SMO svmPoly3 = new SMO();
            PolyKernel poly3 = new PolyKernel();
            poly3.setExponent(3);
            svmPoly3.setKernel(poly3);
            svmPoly3.setC(1.0);
            svmPoly3.buildClassifier(egitim);

            Evaluation evalPoly3 = new Evaluation(egitim);
            evalPoly3.evaluateModel(svmPoly3, test);
            double accPoly3 = evalPoly3.pctCorrect();

            if (accPoly3 > enIyiDogruluk) {
                enIyiDogruluk = accPoly3;
                enIyiKernel = "Polynomial d=3";
            }
            System.out.println("Polynomial d=3 Doğruluk Oranı: " + accPoly3);

            // RBF Kernel
            SMO svmRBF = new SMO();
            RBFKernel rbf = new RBFKernel();
            rbf.setGamma(1.0);
            svmRBF.setKernel(rbf);
            svmRBF.setC(1.0);
            svmRBF.buildClassifier(egitim);

            Evaluation evalRBF = new Evaluation(egitim);
            evalRBF.evaluateModel(svmRBF, test);
            double accRBF = evalRBF.pctCorrect();

            if (accPoly3 > enIyiDogruluk) {
                enIyiDogruluk = accRBF;
                enIyiKernel = "RBF";
            }
            System.out.println("RBF Doğruluk Oranı: " + accRBF);

            System.out.println("\n Secilen Kernel: " + enIyiKernel);

            // RBF Gamma Grid Search
            double[] gammaDegerleri = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
            double enIyiGamma = 1.0;
            enIyiDogruluk = 0;

            for (double gamma : gammaDegerleri) {
                SMO svmGamma = new SMO();
                RBFKernel kernelGamma = new RBFKernel();
                kernelGamma.setGamma(gamma);
                svmGamma.setKernel(kernelGamma);
                svmGamma.setC(1.0);
                svmGamma.buildClassifier(egitim);

                Evaluation evalGamma = new Evaluation(egitim);
                evalGamma.evaluateModel(svmGamma, test);
                double acc = evalGamma.pctCorrect();

                if (acc > enIyiDogruluk) {
                    enIyiDogruluk = acc;
                    enIyiGamma = gamma;
                }
                System.out.println("Gamma: " + gamma + ", Doğruluk Değeri: " + acc);
            }

            System.out.println("Secilen Gamma: " + enIyiGamma);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
