import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

class dataSet {
    String label, word; String[] words;
    int spamFreq = 0; int hamFreq = 0;

    /* for label and msg */
    dataSet(String label, String[] words) {
        this.label = label;
        this.words = words;
    }

    /* for frequency */
    dataSet(String word, int spamFreq, int hamFreq) {
        this.word = word;
        this.spamFreq = spamFreq;
        this.hamFreq = hamFreq;
    }

    void addSpamFreq(int freq) { this.spamFreq = this.spamFreq + freq; } //add value of spam frequency
    void addHamFreq(int freq) { this.hamFreq = this.hamFreq + freq; } //add value of ham frequency
    int getSpamFreq() { return this.spamFreq; } //get value of spam frequency
    int getHamFreq() { return this.hamFreq; } //get value of ham frequency
}

public class naiveBayesClassifier {
    static Set<dataSet> vocab = new HashSet<>(); //for overall vocabulary (after tokenization)
    static Set<dataSet> trainingData = new HashSet<>(); //for training, 70% of data from vocab; to store element in random order
    static Set<dataSet> testData = new HashSet<>(); //for testing, 30% of data from vocab
    static ArrayDeque<dataSet> wordFreq = new ArrayDeque<>(); //ArrayDeque of word frequency: (word, spamFreq, hamFreq)

    static Set<String> vocabDistinct = new HashSet<>(); //unique words in the vocabulary
    static HashSet<Integer> ids = new HashSet<Integer>(); //ids of data

    static int trainLen; //trainLen - size of training data
    static int numSpams = 0; static int numHams = 0; //numSpams - number of spam messages; numHams - number of ham messages
    
    static double probSpam = 0.0; //probability of spam messages
    static double probHam = 0.0; //probability of ham messages
    static double probSpamMsg = 1.0; //P(spam|msg)
    static double probHamMsg = 1.0; //P(ham|msg)

    static int truePositive = 0; static int trueNegative = 0; //truePositive - no. of spam messages classified as spam; trueNegative - no. of ham messages classified as ham
    static int falsePositive = 0; static int falseNegative = 0; //falsePositive - no. of ham messages misclassified as spam; falseNegative - no. of spam messages misclassified as ham
    
    /* FORM VOCABULARY
        - scan and store SMS Spam Collection
            - remove punctuations, convert to lower case, split text into words (separated by space)
        - generate random numbers of ID; use IDs as iterator
        - store dataSet (id, label, words[]) to vocabulary
    */
    static void formVocab(FileInputStream spamData) throws IOException {
        Scanner data = new Scanner(spamData);
        ArrayDeque<String[]> dataToWords = new ArrayDeque<>();
        
        /* scan and store SMS Spam Collection */
        while (data.hasNextLine()) { //read file per line
            //remove punctuations; convert to lower case; split text into words (separated by space)
            dataToWords.add(data.nextLine().toLowerCase() //convert
                .replaceAll("[\t\\p{Punct}]", " ") //remove
                .replaceAll("[^a-zA-Z0-9 ]", "") //convert
                .split("\\s+")); //split
        }

        /* store dataSet to vocabulary vocab */
        Iterator numIterator = dataToWords.iterator();
        while (numIterator.hasNext()) { 
            List<String> sentence = Arrays.asList(dataToWords.remove()); //convert each array to list
            String label = sentence.get(0); //store first word (ham/spam) as label
            String[] words = sentence.stream()
                .filter((txt) -> !label.equals(txt)) //filter txt excluding the label
                .toArray( String[]::new ); //store the words from sentence (excluding label) to array words[]

            dataSet ds = new dataSet(label, words);
            vocab.add(ds);
        }

        data.close();
    }

    /* TRAIN
        - split data for training and test (70/30)
        - store words to trainingDistinct (no duplicate)
        - count frequency of word based on label (spam:ham)
        - compute probability of spam & ham messages
    */
    static void train() throws IOException {
        File file = new File("TrainingData.txt"); //where trainingData will be printed out
        FileWriter output = new FileWriter(file, false);
        
        /* split data for training and test */
        trainLen = (int)(vocab.size() * 0.7); //70% of data is for training
        trainingData = vocab.stream()
            .limit(trainLen) //limit to 70% of data
            .collect( Collectors.toCollection(HashSet::new) ); //store 70% of data from vocab to trainingData
        testData = vocab.stream()
            .skip(trainLen) //skip trainingData
            .limit(vocab.size()) //limit to size of vocab
            .collect( Collectors.toCollection(HashSet::new)); //store remaining data from vocab to testData

        /* store words to vocabDistinct (no duplicate) */
        for (dataSet ds: trainingData) {
            for (String word: ds.words) {
                vocabDistinct.add(word); //vocabDistinct is a hashSet (cannot contain duplicate elements)
            }

            if (ds.label.equals("spam")) { numSpams++; } //increment number of spams
            else { numHams++; } //increment number if hams
        }

        /* count frequency of each word in vocab distinct based on label
            - spamFreq: frequency of word in a spam message 
            - hamFreq: frequncy of word in a ham message
        */
        for (String word: vocabDistinct) {
            dataSet dataFreq = new dataSet(word, 0, 0);

            /* search word in each msg in trainingData */
            for (dataSet ds: trainingData) {
                List<String> trainingWords = Arrays.stream(ds.words).collect(Collectors.toList()); //to easily count freq of word
                int freq = Collections.frequency(trainingWords, word); //count freq of word in each dataSet (msg)

                String label = ds.label;
                switch(label) {
                    case "spam": dataFreq.addSpamFreq(freq); break; //add spam frequency of word 
                    case "ham": dataFreq.addHamFreq(freq); break;//add ham frequency of word
                }
            }

            wordFreq.add(dataFreq);
            output.write(dataFreq.word + " " + dataFreq.getSpamFreq() + " " + dataFreq.getHamFreq() + "\n");
        }

        /* compute statistics */
        probSpam = numSpams / (trainLen + 0.0); //probability of spam messages
        probHam = numHams / (trainLen + 0.0); //probability of ham messages

        output.close();
    }

    /* TEST
        - copy testData to list to easily shuffle data for random testing
        - testData is divided into 5 trials
        - test retrieved data with and without smoothing
            - compute probability of each event
            - at the same time, apply the bayesian theorem
            - if withSmoothing, with every 0 count, a virtual count is added
        - after predicting, check for trueNegative, truePositive, falseNegative, and falsePositive
        - after every last element of each trial, computeAccuracy
        - remove tested data from the list
    */
    static void test() throws IOException {
        File file = new File("TestData.txt"); //where testData will be printed out
        FileWriter output = new FileWriter(file, false);

        File prediction = new File("Prediction.txt"); //where prediction will be printed out
        FileWriter outputP = new FileWriter(prediction, false);
        String labelPrediction;

        /* copy data to list to easily shuffle for random testing */
        List<dataSet> testList = new ArrayList<dataSet>();
        testList.addAll(testData);
        Collections.shuffle(testList);
        int i = 1; //trial number

        /* test testList */
        while (!testList.isEmpty()) {
            int predictLength = Math.round(testData.size()/5) + 1; //compute 1/5 length of testData for the 5 trials
            if (predictLength > testList.size()) { predictLength = testList.size(); } //last remaining data from testList
            List<dataSet> toPredictList = testList.subList(0, predictLength); //pass data from testList from first index until computed predictLength

            Boolean withSmoothing = false;

            /* loop thru same data 2x to test and compare with and without smoothing */
            for (int loop=1; loop <= 2; loop++) {
                if (loop == 2) { withSmoothing = true; } //on the 2nd loop, run test on same toPredictList withSmoothing

                /* loop thru each dataSet in toPredictList */
                for (dataSet ds: toPredictList) {
                    probSpamMsg = probSpam; //set initial value in solving for P(spam|msg)
                    probHamMsg = probHam; //set initial value in solving for P(ham|msg)

                    /* loop thru each word in current dataSet */
                    for (String word: ds.words) {
                        //search through data that matches current word
                        List<dataSet> dataWord = wordFreq.stream()
                            .filter(w -> w.word.equals(word))
                            .collect(Collectors.toList());
                        
                        /* compute statistics */
                        if (dataWord.isEmpty()) {
                            if (withSmoothing) {
                                int spamWord = 0; int hamWord = 0; //spam & ham freq of word
                                probSpamMsg *= smoothing(spamWord, numSpams);
                                probHamMsg *= smoothing(hamWord, numHams);
                            }

                            continue;
                        } else {
                            int spamWord = dataWord.get(0).getSpamFreq(); //get spam freq of word
                            int hamWord = dataWord.get(0).getHamFreq(); //get ham freq of word
                            
                            if (spamWord == 0 && withSmoothing) { probSpamMsg *= smoothing(spamWord, numSpams); } //if value is 0 and withSmoothing is true
                                else { probSpamMsg *= spamWord / (numSpams + 0.0); } //withSmoothing is false, keep value as it is
                            if (hamWord == 0 && withSmoothing) { probHamMsg *= smoothing(hamWord, numHams); } //if value is 0 and withSmoothing is true
                                else { probHamMsg *= hamWord / (numHams + 0.0); } //withSmoothing is false, keep value as it is
                            
                            output.write(spamWord + "/" + numSpams + " : " + hamWord + "/" + numHams + " ");
                        } 
                    } output.write("\n");
                    
                    /* values for trueNegative, truePositive, falseNegative, falsePositive */
                    if (probSpamMsg > probHamMsg) { //prediction: spam
                        labelPrediction = "spam";
                        if (ds.label.equals("spam")) { truePositive++;} //label: spam
                        else { falsePositive++;} //label: ham
                    } else { //prediction: ham
                        labelPrediction = "ham";
                        if (ds.label.equals("ham")) { trueNegative++;} //label: ham
                        else { falseNegative++;} //label: spam
                    }
                    
                    outputP.write("Label: " + ds.label + " Prediction: " + labelPrediction + "\n"); //print prediction (label, prediction)

                    dataSet lastElement = toPredictList.get(toPredictList.size()-1); //retrieve lastElement
                    if (ds == lastElement) { //if last element is tested, computeAccuracy and print output
                        computeAccuracy(i, withSmoothing);
                    }
                }

                truePositive = trueNegative = falsePositive = falseNegative = 0; //reset values
            }
            
            i++;
            testList.subList(0, predictLength).clear(); //remove tested data from list
        }

        output.close();
        outputP.close();
    }

    /* COMPUTE ACCURACY 
        - solve for precision given the truePositive and falsePositive values
        - solve for recall given the truePositive and falseNegative values
        - print data results
     */
    static void computeAccuracy(int trialNumber, boolean withSmoothing) throws IOException {
        File file = new File("Output.txt");
        FileWriter output = new FileWriter(file, true);
        
        double precision = truePositive / (truePositive + falsePositive + 0.0);
        double recall = truePositive / (truePositive + falseNegative + 0.0);

        String s; if (withSmoothing) { s = "Yes"; } else { s = "No"; }
        output.write("\nTrial Number: " + trialNumber + " With Smoothing: " + s + "\n");
        output.write("TP: "+truePositive+" TN: "+trueNegative+" FP: "+falsePositive+" FN: "+falseNegative + "\n");
        output.write("Preicision: "+precision + "\n");
        output.write("Recall: "+ recall + "\n");

        output.close();
    }

    /* SMOOTHING
        - if count is zero, add virtual counts
    */
    static double smoothing(int msg, int num) {
        double prob = (msg + 1.0) / (num + 2.0); 
        return prob;
    }

    public static void main(String[] args) throws Exception {
        FileInputStream dataSet = new FileInputStream("SMSSpamCollection");
        formVocab(dataSet);
        train();
        test();
    }
}
