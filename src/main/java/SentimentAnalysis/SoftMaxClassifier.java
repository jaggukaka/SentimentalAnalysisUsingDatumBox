package SentimentAnalysis;

import com.datumbox.framework.machinelearning.classification.SoftMaxRegression;
import extended.TextClassiferExtendend;

/**
 * Created by jreddypyla on 4/25/15.
 */
public class SoftMaxClassifier extends GenericClassifier {

    @Override
    public void classify(TextClassiferExtendend.TrainingParameters trainingParameters) {

        trainingParameters.setMLmodelClass(SoftMaxRegression.class);
        SoftMaxRegression.TrainingParameters classifierTrainingParameters = new SoftMaxRegression.TrainingParameters();
        trainingParameters.setMLmodelTrainingParameters(classifierTrainingParameters);
        classifierTrainingParameters.setTotalIterations(100);
        classifierTrainingParameters.setLearningRate(0.1);
    }


    public static String stats(TextClassiferExtendend currentInstance) {
        StringBuffer buffer = new StringBuffer(GenericClassifier.stats(currentInstance));
        SoftMaxRegression.ValidationMetrics maValidationMetrics = (SoftMaxRegression.ValidationMetrics)currentInstance
                .getValidationMetrics();
        buffer.append("SSE -- " + maValidationMetrics.getSSE() + "\n");
        buffer.append("R Square Count -- " + maValidationMetrics.getCountRSquare() + "\n");
        return buffer.toString();
    }
}
