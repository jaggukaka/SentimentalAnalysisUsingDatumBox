package SentimentAnalysis;

import com.datumbox.framework.machinelearning.classification.OrdinalRegression;
import extended.TextClassiferExtendend;

/**
 * Created by jreddypyla on 4/25/15.
 */
public class OrdinalClassifier extends GenericClassifier{


    @Override
    public void classify(TextClassiferExtendend.TrainingParameters trainingParameters) {
        trainingParameters.setMLmodelClass(OrdinalRegression.class);
        OrdinalRegression.TrainingParameters classifierTrainingParameters = new OrdinalRegression.TrainingParameters();
        trainingParameters.setMLmodelTrainingParameters(classifierTrainingParameters);
        classifierTrainingParameters.setTotalIterations(100);
    }

    public static String stats(TextClassiferExtendend currentInstance) {
        StringBuffer buffer = new StringBuffer(GenericClassifier.stats(currentInstance));
        OrdinalRegression.ValidationMetrics maValidationMetrics = (OrdinalRegression.ValidationMetrics)currentInstance
                .getValidationMetrics();
        buffer.append("SSE -- " + maValidationMetrics.getSSE() + "\n");
        buffer.append("R Square Count -- " + maValidationMetrics.getCountRSquare() + "\n");
        return buffer.toString();
    }
}
