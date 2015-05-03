package SentimentAnalysis;

import extended.TextClassiferExtendend;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TwitterSentimentAnalysis {
    public static void main(String[] args) throws URISyntaxException {
        TwitterSentimentAnalysis twitterSentimentAnalysis = new TwitterSentimentAnalysis();
        String pos = "file:////Users/jreddypyla/Documents/mystudy/SentimentAnalysis/latestdataset/Positive.csv";
        String neg = "file:////Users/jreddypyla/Documents/mystudy/SentimentAnalysis/latestdataset/Negative.csv";
        String neu = "file:////Users/jreddypyla/Documents/mystudy/SentimentAnalysis/latestdataset/Neutral.csv";
        //String irr = "file:////Users/jreddypyla/Documents/mystudy/SentimentAnalysis/latestdataset/ankithalabelled.csv_irr";
        Scanner scanner = new Scanner(System.in);

        while (scanner.hasNextLine()) {
            ExecutionParams executionParams = twitterSentimentAnalysis.getExecutionParams(scanner.nextLine());
            String command = executionParams.getCommand();
            if ("predict".equalsIgnoreCase(command)) {
                System.out.println(twitterSentimentAnalysis.predict
                        ("file:////Users/jreddypyla/Documents/mystudy/priyanka/pony/tweetscsv.csv",
                                executionParams));
            } else if ("train".equalsIgnoreCase(command)) {
                twitterSentimentAnalysis.train(pos, neg, neu, null, executionParams);
            } else if ("trpr".equalsIgnoreCase(command)) {
                twitterSentimentAnalysis.train(pos, neg, neu, null, executionParams);
                System.out.println(twitterSentimentAnalysis.predict
                        ("file:////Users/jreddypyla/Documents/mystudy/priyanka/pony/tweetscsv.csv",
                                executionParams));
            } else if ("quit".equalsIgnoreCase(command)) {
                break;
            } else {
                System.out.println("Invalid input : " + command);
            }
        }
        scanner.close();
    }

    private  ExecutionParams getExecutionParams(String s) {
        String[] params = s.split(",");

        int numFeatures = -1;
        try {
         numFeatures = Integer.parseInt(params[3]);
        } catch (Exception e) {

        }
        if (params.length == 6) return new ExecutionParams(params[0], params[1], params[2], numFeatures, params[4],
                params[5]);
        else if (params.length == 5) return new ExecutionParams(params[0], params[1], params[2], numFeatures,
                params[4], null);
        else if (params.length == 4) return new ExecutionParams(params[0], params[1], params[2], numFeatures, null,
                null);
        else if (params.length == 3) return new ExecutionParams(params[0], params[1], params[2], numFeatures,
                null,null);
        else if (params.length == 2) return new ExecutionParams(params[0], params[1], null, numFeatures, null, null);
        else return new ExecutionParams(params[0], null, null, numFeatures, null, null);
    }


    public Stats predict(String path, ExecutionParams executionParams) throws URISyntaxException {
        URI datasetURI = new URI(path);
        String classifier = executionParams.toString();
        List<Object> result = SentimentAnalysisHelper.getInstance().getClassifierInstance(executionParams).predict(datasetURI);

        File file = new File("Results_" + classifier + ".txt");
        BufferedWriter bufferedWriter = null;
        try {
            bufferedWriter = new BufferedWriter(new FileWriter(file));
            BufferedReader br = new BufferedReader(new
                    InputStreamReader(new
                    FileInputStream(new File(datasetURI)),
                    "UTF8"));
            int i = 0;
            for (String line; (line = br.readLine()) != null && i < result.size(); ) {

                bufferedWriter.write(line);
                bufferedWriter.write(" -- " + result.get(i));
                bufferedWriter.newLine();
                i++;
            }
            Map<String, Integer> countMap = new HashMap<String, Integer>();
            for (Object o : result) {
                String item = (String) o;
                if (countMap.containsKey(item)) {
                    countMap.put(item, countMap.get(item) + 1);
                } else {
                    countMap.put(item, 1);
                }
            }
            bufferedWriter.newLine();
            bufferedWriter.write("Count Statistics -- " + countMap);
            return new Stats(result, countMap);
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

    private static final class Stats {
        private List<Object> result;
        private Map<String, Integer> countMap;

        public Stats(List<Object> result, Map<String, Integer> countMap) {
            super();
            this.result = result;
            this.countMap = countMap;
        }

        @Override
        public String toString() {
            return "Stats [result=" + result + ", countMap=" + countMap + "]";
        }
    }

    public void train(String positive, String negative, String neutral, String irrelevant,
                      ExecutionParams executionParams) throws URISyntaxException {
        Map<Object, URI> dataset = new HashMap<Object, URI>();
        dataset.put("negative", new URI(negative));
        dataset.put("positive", new URI(positive));
        dataset.put("neutral", new URI(neutral));
        if (irrelevant != null)
            dataset.put("irrelevant", new URI(irrelevant));

        TextClassiferExtendend currentInstance = SentimentAnalysisHelper.getInstance().getClassifierInstance(executionParams);
        currentInstance.train(dataset);
        String classifier = executionParams.toString();


        String stats = SentimentAnalysisHelper.getInstance().getStats(executionParams, currentInstance);
        SentimentAnalysisHelper.getInstance().writeToFile("Stats_" + classifier + ".txt",stats);


        String featureVector = SentimentAnalysisHelper.getInstance().getFeatureVector(currentInstance,
                executionParams);
        SentimentAnalysisHelper.getInstance().writeToFile("FeatureVector_" + classifier + ".txt", featureVector);




    }



    public static final class ExecutionParams {
        private String method;
        private String fs;
        private String normaLizer;
        private String extractor;
        private String command;
        private int numFeatures;

        public ExecutionParams(String command, String method, String fs, int numFeatures, String normaLizer,
                               String extractor
                               ) {
            this.method = method;
            this.fs = fs;
            this.normaLizer = normaLizer;
            this.extractor = extractor;
            this.command = command;
            this.numFeatures = numFeatures;
        }

        public String getMethod() {
            return method;
        }

        public String getFs() {
            return fs;
        }

        public String getNormaLizer() {
            return normaLizer;
        }

        public String getExtractor() {
            return extractor;
        }

        public String getCommand() {
            return command;
        }

        public int getNumFeatures() {
            return numFeatures;
        }

        @Override
        public String toString() {
            return method + "_" + fs + "_" + numFeatures + "_" + normaLizer + "_" + extractor;
        }
    }
}