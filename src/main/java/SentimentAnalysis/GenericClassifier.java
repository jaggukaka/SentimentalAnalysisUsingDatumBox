package SentimentAnalysis;

import com.datumbox.framework.machinelearning.common.bases.mlmodels.BaseMLclassifier;
import extended.TextClassiferExtendend;

/**
 * Created by jreddypyla on 4/25/15.
 */
public abstract class GenericClassifier {

    public abstract void classify(TextClassiferExtendend.TrainingParameters trainingParameters);

    public TextClassiferExtendend textClassifier;

    public static final transient int MAX_FEATURES = 100000;

    public static String stats(TextClassiferExtendend currentInstance) {
        BaseMLclassifier.ValidationMetrics maValidationMetrics = (BaseMLclassifier.ValidationMetrics)currentInstance
                .getValidationMetrics();
        StringBuffer buffer = new StringBuffer();
        buffer.append("Accuracy -- " + maValidationMetrics.getAccuracy()*100 + "\n");
        buffer.append("Contingency Table -- " + maValidationMetrics.getContingencyTable() + "\n");
        buffer.append("Macro F1 -- " + maValidationMetrics.getMacroF1() + "\n");
        buffer.append("Macro Precision -- " + maValidationMetrics.getMacroPrecision() + "\n");
        buffer.append("Macro Recall -- " + maValidationMetrics.getMacroRecall() + "\n");
        buffer.append("Micro F1 -- " + maValidationMetrics.getMicroF1() + "\n");
        buffer.append("Micro Precision -- " + maValidationMetrics.getMicroPrecision() + "\n");
        buffer.append("Micro Recall -- " + maValidationMetrics.getMicroRecall() + "\n");

        return buffer.toString();
    }

    public int getMaxFeatures(String fs) {
        return MAX_FEATURES;
    }


}
