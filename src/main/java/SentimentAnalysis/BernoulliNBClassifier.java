package SentimentAnalysis;

import com.datumbox.framework.machinelearning.classification.BernoulliNaiveBayes;
import extended.TextClassiferExtendend;

/**
 * Created by jreddypyla on 4/25/15.
 */
public class BernoulliNBClassifier extends GenericClassifier {

    @Override
    public void classify(TextClassiferExtendend.TrainingParameters trainingParameters) {

        trainingParameters.setMLmodelClass(BernoulliNaiveBayes.class);
        BernoulliNaiveBayes.TrainingParameters classifierTrainingParameters = new BernoulliNaiveBayes.TrainingParameters();
        trainingParameters.setMLmodelTrainingParameters(classifierTrainingParameters);
        classifierTrainingParameters.setMultiProbabilityWeighted(false);
    }


}
