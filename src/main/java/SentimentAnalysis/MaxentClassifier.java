package SentimentAnalysis;

import com.datumbox.framework.machinelearning.classification.MaximumEntropy;
import extended.TextClassiferExtendend;

/**
 * Created by jreddypyla on 4/25/15.
 */
public class MaxentClassifier extends GenericClassifier {


    @Override
    public void classify(TextClassiferExtendend.TrainingParameters trainingParameters) {

        trainingParameters.setMLmodelClass(MaximumEntropy.class);
        MaximumEntropy.TrainingParameters classifierTrainingParameters = new MaximumEntropy.TrainingParameters();
        trainingParameters.setMLmodelTrainingParameters(classifierTrainingParameters);
        classifierTrainingParameters.setTotalIterations(100);
    }


}
