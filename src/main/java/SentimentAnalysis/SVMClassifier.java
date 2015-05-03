package SentimentAnalysis;

import com.datumbox.framework.machinelearning.classification.SupportVectorMachine;
import extended.TextClassiferExtendend;
import libsvm.svm_parameter;

/**
 * Created by jreddypyla on 4/25/15.
 */
public class SVMClassifier extends GenericClassifier {

    @Override
    public void classify(TextClassiferExtendend.TrainingParameters trainingParameters) {

        trainingParameters.setMLmodelClass(SupportVectorMachine.class);
        SupportVectorMachine.TrainingParameters classifierTrainingParameters = new SupportVectorMachine.TrainingParameters();
        classifierTrainingParameters.getSvmParameter().kernel_type = svm_parameter.RBF;
        classifierTrainingParameters.getSvmParameter().C = 1;
        classifierTrainingParameters.getSvmParameter().gamma = 0.10000000000000001;
        classifierTrainingParameters.getSvmParameter().cache_size = 2000;
        trainingParameters.setMLmodelTrainingParameters(classifierTrainingParameters);

    }


    public int getMaxFeatures(String fs) {
        return 2000;
    }
}
