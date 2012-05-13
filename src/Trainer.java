import java.awt.Image;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.stream.JsonWriter;

import de.pribluda.android.jsonmarshaller.JSONMarshaller;

/**
 * sample application demonstrating training of matchers and creation of
 * data sets for recognition.  reads samples prepared by sample demo application.
 *
 * @author Konstantin Pribluda
 */
public class Trainer {
    // constant defining possible character set - only those characters woll be trained
    // and recognized later
    private static final String DAT_SOURCE_DIR = "/home/kevin/ocrsamples/dats";
    private static final String FREESPACES_OUTPUT = "/home/kevin/ocrsamples/output/freespaces.json";
    private static final String MOMENTS_OUTPUT = "/home/kevin/ocrsamples/output/moments.json";
    public static final String POSSIBLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ8th0e9quickbr1235own4"; 
    
    private static final String TAB = "\t";

    /**
     * @param args args[0] contains directory witj sample files
     */
    public static void main(String[] args) throws IOException, InvocationTargetException, NoSuchMethodException, IllegalAccessException {
        File samples = new File(DAT_SOURCE_DIR);
        if (!samples.exists() || !samples.isDirectory()) {
            System.err.println(args[0] + "must be directory");
            return;
        }

        /**
         * extractor(s) used here shall be the same as used for recognition 
         */
        FeatureExtractor extractor = new HuMoments();

        /**
         * free spaces extracts amount ot continuous free spaces in a glyph,
         * and is suitable to boost recognition quality
         */
        FreeSpacesExtractor freeSpacesExtractor = new FreeSpacesExtractor();
        FreeSpacesMatcher freeSpacesMatcher = new FreeSpacesMatcher();

        MetricMatcher metricMatcher = new MetricMatcher();

        // collect  sample quality data
        List<String> files = new ArrayList();
        Map<String, Integer> errors = new HashMap();


        // list of characters
        List<Character> characters = new ArrayList();
        // List of computed moments
        List<double[]> moments = new ArrayList();
        List<double[]> freeSpaces = new ArrayList();

        //  character features will be used to create indivudual clusters 
        Map<Character, List<double[]>> characterFeatures = new HashMap();
        for (Character c : POSSIBLE.toCharArray()) {
            characterFeatures.put(c, new ArrayList<double[]>());
        }

        /**
         * loop through sample files
         */

        for (File sample : samples.listFiles()) {
            if (!sample.isFile() || !sample.canRead()) {
                System.err.println("skipping unreadable file: " + sample);
                continue;
            }

            // extract characters from  filegg name
            final String expected = sample.getName().split("_")[0];
            List<Image> images = loadSamples(sample);

            // amount of samples shall match expected characters
            if (expected.length() != images.size()) {
                System.err.println("file " + sample + " content does not match name:" + images.size());
                continue;
            }


            // extract features from glyphs and store them for later use
            System.out.println("processing file:" + sample);
            for (int i = 0; i < images.size(); i++) {
                char c = expected.charAt(i);
                characters.add(c);
                // extract moments
                double[] extractedMoments = extractor.extract(images.get(i));
                double[] extractedFreeSpaces = freeSpacesExtractor.extract(images.get(i));

                // also train freespaces matcher
                freeSpaces.add(extractedFreeSpaces);
                freeSpacesMatcher.train(c, (int) Math.round(extractedFreeSpaces[0]));

                moments.add(extractedMoments);
                characterFeatures.get(c).add(extractedMoments);
            }


        }


        // samples are extracted
        System.out.println("processed" + characters.size() + " glyphs");


        // merge and teach clusters
        for (Character c : POSSIBLE.toCharArray()) {
            System.err.println("training cluster for character:" + c);
            final List<double[]> initialDoubles = characterFeatures.get(c);

            // train cluster
            MahalanobisDistanceCluster cluster = new MahalanobisDistanceCluster(extractor.getSize());
            for (double[] features : initialDoubles) {
                cluster.train(features);
            }

            // compute distances
            ArrayList<Double> distances = new ArrayList();
            for (double[] features : initialDoubles) {
                distances.add(cluster.distance(features));
            }
            Collections.sort(distances);

            final double red = distances.get(distances.size() - 1);
            final double yellow = red * 0.8;
            System.err.println("distance yellow: " + yellow + " red: " + red);

            metricMatcher.addMetric(cluster, c, yellow, red);
        }


        // perform image recognition to demonstrate sample quality
        // (same process as in real rimage recognition
        int correctChars = 0;
        int errorCount = 0;
        for (int i = 0; i < characters.size(); i++) {

            final char expectedChar = characters.get(i);
            // moment vector to be matched
            final double[] vector = moments.get(i);


            final List<Match> matchesFromCluster = metricMatcher.classify(vector);
            final List<Match> matchesFromFreeSpaces = freeSpacesMatcher.classify(freeSpaces.get(i));

            final List<Match> mergedList = MatcherUtil.merge(matchesFromCluster, matchesFromFreeSpaces);

            final Match match = mergedList.get(0);
            final Character result = match.getChr();
            if (result == match.getChr()) {
                correctChars++;
            } else {
                errorCount++;

                System.out.println(expectedChar + " ||  " + result);

                System.out.println("");


                System.out.println("Clusters:");
                printMatchList(matchesFromCluster);
                System.out.println("---------------------------");
                System.out.println("Free spaces:");
                printMatchList(matchesFromFreeSpaces);
                System.out.println("---------------------------");
            }
        }

        System.out.println("correct: " + correctChars + " errors:" + errorCount);


        // write configuration JSON to standart output

        //   save out data
        System.out.println("------------------[ free ]--------------------");

        // TODO:  maybe this logic belongs into matcher.

//        writeJsonArray(freeSpacesMatcher.getContainers());
        writeJsonFile(freeSpacesMatcher.getContainers(), FREESPACES_OUTPUT);

        System.out.println("----------------[ free ends ]-----------------");

        System.out.println("----------[ cluster data ]----------------");

        final List<MetricContainer> metricContainerList = metricMatcher.containers();
        writeJsonFile(metricContainerList, MOMENTS_OUTPUT);
//        writeJsonArray(metricContainerList);

        System.out.println("----------------------------------------");

    }


    /**
     * load sample images from samples,  samples contains sliced and shrinked glyphs
     *
     * @param samples file containing image samples
     * @return list of glyphs
     */
    public static List<Image> loadSamples(File samples) throws IOException {
        List<Image> glyphs = new ArrayList();


        DataInputStream dis = new DataInputStream(new FileInputStream(samples));

        //read amount samples contained here
        int amountSamples = dis.readInt();
        // and images
        for (int i = 0; i < amountSamples; i++) {
            int width = dis.readInt();
            int height = dis.readInt();

            int data[] = new int[width * height];
            for (int j = 0; j < data.length; j++) {
                data[j] = dis.readInt();
            }
            glyphs.add(new PixelImage(data, width, height));
        }

        return glyphs;
    }


    private static void printMatchList(List<Match> matchesFromFreeSpaces) {
        StringBuilder chr;
        StringBuilder dist;
        StringBuilder red;
        chr = new StringBuilder();
        dist = new StringBuilder();
        red = new StringBuilder();
        for (Match m : matchesFromFreeSpaces) {
            chr.append(m.getChr()).append(TAB);
            dist.append(m.getDistance()).append(TAB);
            red.append(m.getRed()).append(TAB);
        }
        System.out.println(chr.toString());
        System.out.println(dist.toString());
        System.out.println(red.toString());
    }


//    private static void writeJsonArray(List list) throws InvocationTargetException, NoSuchMethodException, IllegalAccessException, IOException {
//        JsonWriter writer = new JsonWriter(new OutputStreamWriter(System.out));
//        writer.setIndent("    ");
//        JSONMarshaller.marshallArray(writer, list.toArray());
//        writer.flush();
//    }
    
    private static void writeJsonFile(List list, String filedir) throws InvocationTargetException, NoSuchMethodException, IllegalAccessException, IOException {
    	File newFile = new File(filedir);
    	FileOutputStream myFileWriter =  new FileOutputStream(newFile);
    	JsonWriter writer = new JsonWriter(new OutputStreamWriter(myFileWriter));
      writer.setIndent("    ");
      JSONMarshaller.marshallArray(writer, list.toArray());
      writer.flush();
    }

}
