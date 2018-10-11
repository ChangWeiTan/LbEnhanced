/* Copyright (C) 2018 Chang Wei Tan, Francois Petitjean, Geoff Webb

 This file is part of LbEnhanced.

 LbEnhanced is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, version 3 of the License.

 LbEnhanced is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with LbEnhanced.  If not, see <http://www.gnu.org/licenses/>. */
package experiments;

import classifiers.*;
import utilities.DataLoader;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Pruning Power and Classification Time with Random Sampling
 * java -Xmx14g -Xms14g -cp $LIBDIR: experiments.PruneAndTimeRandomSampling $PROJECTDIR $DATASETDIR $PROBLEM $RUNS $WIN $TOESTIMATEORNOT $LOWERBOUND $ENHANCEDPARAM
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class PruneAndTimeRandomSampling extends Experiments {
    private static int enhancedParam;
    private static String lowerBound;

    public static void main(String[] args) {
        final long heapSize = Runtime.getRuntime().totalMemory();
        final long heapMaxSize = Runtime.getRuntime().maxMemory();
        final long heapFreeSize = Runtime.getRuntime().freeMemory();
        System.out.println(String.format("[PRUNED-TIME] Heap:      %20d", heapSize));
        System.out.println(String.format("[PRUNED-TIME] Max Heap:  %20d", heapMaxSize));
        System.out.println(String.format("[PRUNED-TIME] Free Heap: %20d", heapFreeSize));

        problem = "Adiac";
        nbRuns = 5;
        enhancedParam = 4;
        lowerBound = "LbSuperEnhanced2";
        r = -1;
        window = -1;
        boolean estimate = false;

        doNothing();

        if (args.length >= 1) projectPath = args[0];
        if (args.length >= 2) datasetPath = args[1];
        if (args.length >= 3) problem = args[2];
        if (args.length >= 4) nbRuns = Integer.parseInt(args[3]);
        if (args.length >= 5) {
            System.out.println(args[4].contains("."));
            if (args[4].contains("."))
                r = Double.parseDouble(args[4]);
            else
                window = Integer.parseInt(args[4]);
        }
        if (args.length >= 6) estimate = Boolean.parseBoolean(args[5]);
        if (args.length >= 7) lowerBound = args[6];
        if (args.length >= 8) enhancedParam = Integer.parseInt(args[7]);

        System.out.println("[PRUNED-TIME] Input Arguments:");
        System.out.println(String.format("[PRUNED-TIME] Project Path:   %s", projectPath));
        System.out.println(String.format("[PRUNED-TIME] Dataset Path:   %s", datasetPath));
        System.out.println(String.format("[PRUNED-TIME] Problem:        %s", problem));
        System.out.println(String.format("[PRUNED-TIME] Runs:           %d", nbRuns));
        if (r >= 0)
            System.out.println(String.format("[PRUNED-TIME] r:              %.2f", r));
        else
            System.out.println(String.format("[PRUNED-TIME] window:         %d", window));
        System.out.println(String.format("[PRUNED-TIME] Estimate:       %B", estimate));
        System.out.println(String.format("[PRUNED-TIME] Lower Bound:    %s", lowerBound));
        System.out.println(String.format("[PRUNED-TIME] Param:          %d", enhancedParam));
        System.out.println("");

        singleProblem(problem, estimate);
    }

    private static void singleProblem(String problem, boolean estimate) {
        System.out.println("[PRUNED-TIME] Loading " + problem);
        final Instances train = DataLoader.loadTrain(problem, datasetPath);
        final Instances test = DataLoader.loadTest(problem, datasetPath);
        System.out.println(String.format("[PRUNED-TIME] %s loaded", problem));

        final int seqLen = train.numAttributes() - 1;
        if (r < 0)
            r = 1.0 * window / seqLen;
        else if (window < 0)
            window = (int) (r * seqLen);

        double accuracy, elapsedTime, pruned;
        DTW1NN classifier;
        System.out.println(String.format("[PRUNED-TIME] Creating %s classifier", lowerBound));
        switch (lowerBound) {
            case "LbKim":
                classifier = new LbKimDTW1NN();
                break;
            case "LbKeogh":
                classifier = new LbKeoghDTW1NN();
                break;
            case "LbImproved":
                classifier = new LbImprovedDTW1NN();
                break;
            case "LbNew":
                classifier = new LbNewDTW1NN();
                break;
            case "LbEnhanced":
                classifier = new LbEnhancedDTW1NN();
                classifier.setV(enhancedParam);
                break;
            case "Naive":
            default:
                classifier = new DTW1NN();
                break;
        }

        String filename;
        setResDir("out_sdm/prune_time_random/" + problem + "/");
        if (lowerBound.equals("LbEnhanced"))
            filename = resDir + problem + "_" + lowerBound + enhancedParam + ".csv";
        else
            filename = resDir + problem + "_" + lowerBound + ".csv";

        for (int run = 0; run < nbRuns; run++) {
            append = window != 1 || run != 0;
            System.out.print(String.format("[PRUNED-TIME] Run %d, Window size %d (%.2f)", run, window, r));
            train.randomize(new Random(run));
            test.randomize(new Random(run));

            classifier.init(train, test, window);

            if (!estimate) {
                classifier.startTime = System.nanoTime();
                accuracy = classifier.accuracy();
                classifier.stopTime = System.nanoTime();
                elapsedTime = 1.0 * (classifier.stopTime - classifier.startTime);
                pruned = classifier.getPruned();
            } else {
                classifier.startTime = System.nanoTime();
                accuracy = classifier.accuracyEstimate();
                classifier.stopTime = System.nanoTime();
                elapsedTime = 1.0 * (classifier.stopTime - classifier.startTime);
                pruned = classifier.getPruned();
                if (classifier.OVERTIME) {
                    elapsedTime *= test.numInstances() / (classifier.queryIndex + 1);
                    accuracy = 2;
                    pruned = 2;
                }
            }

            if (lowerBound.equals("LbEnhanced"))
                System.out.println(String.format(" -- %s%d: %3d s %10d ns -- %.5f -- %6.3f",
                        lowerBound, enhancedParam, (int) (elapsedTime / 1e9), (int) (elapsedTime % 1e9), accuracy, pruned));
            else
                System.out.println(String.format(" -- %s: %3d s %10d ns -- %.5f -- %6.3f",
                        lowerBound, (int) (elapsedTime / 1e9), (int) (elapsedTime % 1e9), accuracy, pruned));

            save(filename, run, window, r, elapsedTime, accuracy, pruned, append);
        }

        System.out.println("");
    }


    private static void save(String filename, int run, int win, double r, double time, double accuracy, double pruned, boolean append) {
        FileWriter out;
        final String comma = ",";
        try {
            out = new FileWriter(filename, append);
            if (!append) out.append("Run,r,Win,Time,Accuracy,PrunePower\n");
            out.append(String.valueOf(run)).append(comma)
                    .append(String.valueOf(r)).append(comma)
                    .append(String.valueOf(win)).append(comma)
                    .append(String.valueOf(time / 1e9)).append(comma)
                    .append(String.valueOf(accuracy)).append(comma)
                    .append(String.valueOf(pruned)).append("\n");
            out.flush();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
