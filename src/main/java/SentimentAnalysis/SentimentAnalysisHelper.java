package SentimentAnalysis;

import com.datumbox.common.persistentstorage.factories.InMemoryStructureFactory;
import com.datumbox.common.utilities.RandomValue;
import com.datumbox.configuration.MemoryConfiguration;
import com.datumbox.framework.machinelearning.common.bases.featureselection.CategoricalFeatureSelection;
import com.datumbox.framework.machinelearning.datatransformation.DummyXYMinMaxNormalizer;
import com.datumbox.framework.machinelearning.featureselection.categorical.ChisquareSelect;
import com.datumbox.framework.machinelearning.featureselection.categorical.MutualInformation;
import com.datumbox.framework.machinelearning.featureselection.scorebased.TFIDF;
import com.datumbox.framework.utilities.text.extractors.NgramsExtractor;
import extended.TextClassiferExtendend;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


/**
 * Created by jreddypyla on 4/26/15.
 */
public class SentimentAnalysisHelper {


    private SentimentAnalysisHelper() {

        RandomValue.randomGenerator = new Random(42);
        this.memoryConfiguration = new MemoryConfiguration();
        //the analysis is VERY slow if not performed in memory training, so we force it anyway.
        memoryConfiguration.setMapType(InMemoryStructureFactory.MapType.HASH_MAP);

    }

    private static final class SentimentAnalysisHelperHolder {
        private static final SentimentAnalysisHelper SENTIMENT_ANALYSIS_HELPER = new SentimentAnalysisHelper();

    }

    public static SentimentAnalysisHelper getInstance() {
        return SentimentAnalysisHelperHolder.SENTIMENT_ANALYSIS_HELPER;
    }

    private MemoryConfiguration memoryConfiguration;
    private static final transient String BASE_DB_NAME = "TwitterSentimentAnalysis";
    private Map<String, TextClassiferExtendend> TextClassiferExtendendMap = new HashMap<String, TextClassiferExtendend>();

    public TextClassiferExtendend getClassifierInstance(TwitterSentimentAnalysis.ExecutionParams executionParams) {
        String key = executionParams.getMethod() + "_" + executionParams.getFs();

        if (TextClassiferExtendendMap.containsKey(key)) {
            return TextClassiferExtendendMap.get(key);
        }

        GenericClassifier genericClassifier = getLocalClassifier(executionParams);
        TextClassiferExtendend instance = new TextClassiferExtendend(BASE_DB_NAME);
        TextClassiferExtendend.TrainingParameters trainingParameters = instance.getEmptyTrainingParametersObject();
        trainingParameters.setkFolds(5);
        genericClassifier.classify(trainingParameters);

//        trainingParameters.setDataTransformerClass(DummyXYMinMaxNormalizer.class);
//        trainingParameters.setDataTransformerTrainingParameters(new DummyXYMinMaxNormalizer.TrainingParameters());

        String fs = executionParams.getFs();
        int maxFeatures = executionParams.getNumFeatures();
        if (maxFeatures == -1) {
            maxFeatures = genericClassifier.getMaxFeatures(fs);
        }
        setFeatureSelector(fs, trainingParameters, maxFeatures);
        trainingParameters.setTextExtractorClass(NgramsExtractor.class);
        trainingParameters.setTextExtractorTrainingParameters(new NgramsExtractor.Parameters());
        instance.initializeTrainingConfiguration(memoryConfiguration, trainingParameters);

        genericClassifier.textClassifier = instance;
        TextClassiferExtendendMap.put(key, instance);

        return instance;
    }

    private GenericClassifier getLocalClassifier(TwitterSentimentAnalysis.ExecutionParams executionParams) {

        String method = executionParams.getMethod();

        if ("bern".equalsIgnoreCase(method)) {
            return new BernoulliNBClassifier();
        } else if ("svm".equalsIgnoreCase(method)) {
            return new SVMClassifier();
        } else if ("maxent".equalsIgnoreCase(method)) {
            return new MaxentClassifier();
        } else if ("soft".equalsIgnoreCase(method)) {
            return new SoftMaxClassifier();
        } else if ("ord".equalsIgnoreCase(method)) {
            return new OrdinalClassifier();
        }

        return new MaxentClassifier();
    }

    public String getStats(TwitterSentimentAnalysis.ExecutionParams executionParams, TextClassiferExtendend TextClassiferExtendend) {


        String method = executionParams.getMethod();

        if ("bern".equalsIgnoreCase(method)) {
            return BernoulliNBClassifier.stats(TextClassiferExtendend);
        } else if ("svm".equalsIgnoreCase(method)) {
            return SVMClassifier.stats(TextClassiferExtendend);
        } else if ("maxent".equalsIgnoreCase(method)) {
            return MaxentClassifier.stats(TextClassiferExtendend);
        } else if ("soft".equalsIgnoreCase(method)) {
            return SoftMaxClassifier.stats(TextClassiferExtendend);
        } else if ("ord".equalsIgnoreCase(method)) {
            return OrdinalClassifier.stats(TextClassiferExtendend);
        }

        return MaxentClassifier.stats(TextClassiferExtendend);
    }


    public void setFeatureSelector(String fs, TextClassiferExtendend.TrainingParameters trainingParameters, int maxFeatures) {



        if ("chi".equalsIgnoreCase(fs)) {
            trainingParameters.setFeatureSelectionClass(ChisquareSelect.class);
            ChisquareSelect.TrainingParameters fsParams = new ChisquareSelect.TrainingParameters();
            fsParams.setALevel(0.05);
            fsParams.setIgnoringNumericalFeatures(false);
            fsParams.setMaxFeatures(maxFeatures);
            fsParams.setRareFeatureThreshold(10);
            trainingParameters.setFeatureSelectionTrainingParameters(fsParams);
        } else if ("tfidf".equalsIgnoreCase(fs)) {
            setTIIDF(trainingParameters, maxFeatures);
        } else if ("mutual".equalsIgnoreCase(fs)) {
            trainingParameters.setFeatureSelectionClass(MutualInformation.class);
            MutualInformation.TrainingParameters fsParams = new MutualInformation.TrainingParameters();
            fsParams.setIgnoringNumericalFeatures(true);
            fsParams.setMaxFeatures(maxFeatures);
            fsParams.setRareFeatureThreshold(10);
            trainingParameters.setFeatureSelectionTrainingParameters(fsParams);
            trainingParameters.setFeatureSelectionTrainingParameters(fsParams);
        } else {
            //Defaulting to TFIDF
            setTIIDF(trainingParameters, maxFeatures);
        }
    }

    private void setTIIDF(TextClassiferExtendend.TrainingParameters trainingParameters, int maxFeatures) {
        trainingParameters.setFeatureSelectionClass(TFIDF.class);
        TFIDF.TrainingParameters fsParams = new TFIDF.TrainingParameters();
        fsParams.setBinarized(false);
        fsParams.setMaxFeatures(maxFeatures);
        trainingParameters.setFeatureSelectionTrainingParameters(fsParams);
    }

    public String getFeatureVector(TextClassiferExtendend textClassiferExtendend,
                                   TwitterSentimentAnalysis.ExecutionParams executionParams) {

        String fs = executionParams.getFs();
        StringBuffer buffer = new StringBuffer();

        int getTotalFeatures = textClassiferExtendend.getRawFeaturesSize();

        buffer.append("\n ###### Raw set of feaures before feature selection algorithm is applied -- " +
                getTotalFeatures);

        if ("chi".equalsIgnoreCase(fs)) {


            ChisquareSelect.ModelParameters modelParameters = (ChisquareSelect.ModelParameters) textClassiferExtendend.getFeatureSelection()
                    .getModelParameters();

            buffer.append("\n ###### Feature Class Counts ####### \n");
            buffer.append(stringFromMap(modelParameters.getFeatureClassCounts()));
            buffer.append("\n ###### Feature  Counts ####### \n");
            buffer.append(stringFromMap(modelParameters.getFeatureCounts()));
            buffer.append("\n ###### Feature Scores ####### \n");
            buffer.append(stringFromMap(modelParameters.getFeatureScores()));
            buffer.append("\n ###### Class Counts ####### \n");
            buffer.append(stringFromMap(modelParameters.getClassCounts()));
            return buffer.toString();
        } else if ("tfidf".equalsIgnoreCase(fs)) {
            return stringFromMap(((TFIDF.ModelParameters) textClassiferExtendend.getFeatureSelection()
                    .getModelParameters()).getMaxTFIDFfeatureScores());
        } else if ("mutual".equalsIgnoreCase(fs)) {

            MutualInformation.ModelParameters modelParameters = (MutualInformation.ModelParameters) textClassiferExtendend
                    .getFeatureSelection()
                    .getModelParameters();
            buffer.append("\n ###### Feature Class Counts ####### \n");
            buffer.append(stringFromMap(modelParameters.getFeatureClassCounts()));
            buffer.append("\n ###### Feature  Counts ####### \n");
            buffer.append(stringFromMap(modelParameters.getFeatureCounts()));
            buffer.append("\n ###### Feature Scores ####### \n");
            buffer.append(stringFromMap(modelParameters.getFeatureScores()));
            buffer.append("\n ###### Class Counts ####### \n");
            buffer.append(stringFromMap(modelParameters.getClassCounts()));

            return buffer.toString();
        } else {
            return stringFromMap(((TFIDF.ModelParameters) textClassiferExtendend.getFeatureSelection()
                    .getModelParameters()).getMaxTFIDFfeatureScores());
        }

    }

    private String stringFromMap(Map<?, ?> featureScores) {

        StringBuffer buffer = new StringBuffer();
        for (Map.Entry entry : featureScores.entrySet()) {
            buffer.append(entry.getKey());
            buffer.append("--");
            buffer.append(entry.getValue());
            buffer.append('\n');

        }

        return buffer.toString();

    }


    public void writeToFile(String fileName, String value) {

        File file = new File(fileName);
        BufferedWriter bufferedWriter = null;
        try {
            bufferedWriter = new BufferedWriter(new FileWriter(file));
            bufferedWriter.write(value);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        } finally {
            try {
                bufferedWriter.flush();
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


}
