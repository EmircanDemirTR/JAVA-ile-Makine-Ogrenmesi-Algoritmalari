package com.btkakademi.ml.bolum12;


import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.InputStream;
import java.util.Random;

public class WekaPCA {
    static void main() {
        try {
            InputStream is = WekaPCA.class.getClassLoader().getResourceAsStream("datasets/wine.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Standardizasyon
            Standardize standardize = new Standardize();
            standardize.setInputFormat(veri);
            Instances veriStd = Filter.useFilter(veri, standardize);

            // Baseline model
            J48 tree = new J48();
            tree.setConfidenceFactor(0.25f);

            Evaluation evalOrijinal = new Evaluation(veri);
            evalOrijinal.crossValidateModel(tree, veri, 10, new Random(42));

            System.out.println("Orijinal 13 Ozellikli Model: " + evalOrijinal.pctCorrect() + "Doğruluk ve " + evalOrijinal.kappa() + "Kappa");

            // PCA ile Boyut İndirgeme

            double[] varyansDegerleri = {0.80, 0.90, 0.95, 0.99};

            for (double varyans : varyansDegerleri) {
                PrincipalComponents pcaEval = new PrincipalComponents();
                pcaEval.setVarianceCovered(varyans);
                pcaEval.setCenterData(true);

                Ranker ranker = new Ranker();
                ranker.setNumToSelect(-1);

                AttributeSelection pcaFilter = new AttributeSelection();
                pcaFilter.setEvaluator(pcaEval);
                pcaFilter.setSearch(ranker);

                FilteredClassifier fc = new FilteredClassifier();
                J48 fcTree = new J48();
                fcTree.setConfidenceFactor(0.25f);
                fc.setClassifier(fcTree);
                fc.setFilter(pcaFilter);

                Evaluation evalPCA = new Evaluation(veriStd);
                evalPCA.crossValidateModel(fc, veriStd, 10, new Random(42));

                AttributeSelection pcaForCount = new AttributeSelection();
                PrincipalComponents pcaEvalCount = new PrincipalComponents();
                pcaEvalCount.setVarianceCovered(varyans);
                pcaEvalCount.setCenterData(true);
                Ranker rankerCount = new Ranker();
                rankerCount.setNumToSelect(-1);
                pcaForCount.setEvaluator(pcaEvalCount);
                pcaForCount.setSearch(rankerCount);
                pcaForCount.setInputFormat(veriStd);
                Instances pcaVeriCount = Filter.useFilter(veriStd, pcaForCount);

                System.out.println(varyans * 100 + "Varyans   " + (pcaVeriCount.numAttributes() - 1) + "Bilesen"
                        + evalPCA.pctCorrect() + "Dogruluk + " + "        " + evalPCA.kappa() + "Kappa");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
