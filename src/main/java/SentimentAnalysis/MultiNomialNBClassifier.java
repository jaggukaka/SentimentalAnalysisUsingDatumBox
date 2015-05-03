package SentimentAnalysis;

import com.datumbox.framework.machinelearning.classification.MultinomialNaiveBayes;
import extended.TextClassiferExtendend;

/**
 * Created by jreddypyla on 4/25/15.
 */
public class MultiNomialNBClassifier extends GenericClassifier {
    @Override
    public void classify(TextClassiferExtendend.TrainingParameters trainingParameters) {
        trainingParameters.setMLmodelClass(MultinomialNaiveBayes.class);
        MultinomialNaiveBayes.TrainingParameters classifierTrainingParameters = new MultinomialNaiveBayes.TrainingParameters();
        trainingParameters.setMLmodelTrainingParameters(classifierTrainingParameters);
    }


}
