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

import elasticDistances.DTW;
import lowerBounds.*;
import sequences.SequenceStatsCache;
import utilities.DataLoader;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Lower bound compute time vs tightness for UCR datasets
 * java -Xmx14g -Xms14g -cp $LIBDIR: experiments.LbComputeTime $PROJECTDIR $DATASETDIR $PROBLEM $RUNS $WIN $LOWERBOUND $ENHANCEDPARAM
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbComputeTime extends Experiments {
    private static int enhancedParam;
    private static String lowerBound;

    public static void main(String[] args) {
        final long heapSize = Runtime.getRuntime().totalMemory();
        final long heapMaxSize = Runtime.getRuntime().maxMemory();
        final long heapFreeSize = Runtime.getRuntime().freeMemory();
        System.out.println(String.format("[LB-COMPUTE-TIME] Heap:      %20d", heapSize));
        System.out.println(String.format("[LB-COMPUTE-TIME] Max Heap:  %20d", heapMaxSize));
        System.out.println(String.format("[LB-COMPUTE-TIME] Free Heap: %20d", heapFreeSize));

        problem = "Adiac";
        nbRuns = 5;
        enhancedParam = 4;
        lowerBound = "LbEnhanced";
        r = -1;
        window = -1;

        doNothing();

        if (args.length >= 1) projectPath = args[0];
        if (args.length >= 2) datasetPath = args[1];
        if (args.length >= 3) problem = args[2];
        if (args.length >= 4) nbRuns = Integer.parseInt(args[3]);
        if (args.length >= 5) {
            if (args[4].contains("."))
                r = Double.parseDouble(args[4]);
            else
                window = Integer.parseInt(args[4]);
        }
        if (args.length >= 6) lowerBound = args[5];
        if (args.length >= 7) enhancedParam = Integer.parseInt(args[6]);

        System.out.println("[LB-COMPUTE-TIME] Input Arguments:");
        System.out.println(String.format("[LB-COMPUTE-TIME] Project Path:   %s", projectPath));
        System.out.println(String.format("[LB-COMPUTE-TIME] Dataset Path:   %s", datasetPath));
        System.out.println(String.format("[LB-COMPUTE-TIME] Problem:        %s", problem));
        System.out.println(String.format("[LB-COMPUTE-TIME] Runs:           %d", nbRuns));
        if (r >= 0)
            System.out.println(String.format("[LB-COMPUTE-TIME] r:              %.2f", r));
        else if (window >= 0)
            System.out.println(String.format("[LB-COMPUTE-TIME] window:         %d", window));
        else {
            window = 1;
            System.out.println(String.format("[LB-COMPUTE-TIME] window:         %d", window));
        }
        System.out.println(String.format("[LB-COMPUTE-TIME] Lower Bound:    %s", lowerBound));
        System.out.println(String.format("[LB-COMPUTE-TIME] Param:          %d", enhancedParam));
        System.out.println("");

        singleProblem(problem);
    }

    private static void singleProblem(String problem) {
        System.out.println("[LB-COMPUTE-TIME] Loading " + problem);
        final Instances train = DataLoader.loadTrain(problem, datasetPath);
        final Instances test = DataLoader.loadTest(problem, datasetPath);
        System.out.println(String.format("[LB-COMPUTE-TIME] %s loaded", problem));

        final int seqLen = train.numAttributes() - 1;
        if (r < 0)
            r = 1.0 * window / seqLen;
        else if (window < 0)
            window = (int) (r * seqLen);

        long startTime = 0, stopTime = 0;
        double dist = 0;

        DTW dtwComputer = new DTW();
        dtwComputer.setWindowSize(window);

        LbKim kimComputer = new LbKim();
        LbKeogh keoghComputer = new LbKeogh();
        LbImproved improvedComputer = new LbImproved();
        LbNew newComputer = new LbNew();
        LbEnhanced enhancedComputer = new LbEnhanced();

        String filename;
        setResDir("out_sdm/lb_compute_time/" + problem + "/");
        if (lowerBound.equals("LbEnhanced"))
            filename = resDir + problem + "_" + lowerBound + enhancedParam + ".csv";
        else
            filename = resDir + problem + "_" + lowerBound + ".csv";

        double elapsedTime, tightness;
        int comparisons;
        int nTrain = Math.min(train.numInstances(), 100);
        int nTest = Math.min(test.numInstances(), 100);
        double[][] elapsedTimes = new double[nTest][nTrain];
        double[][] tightnesss = new double[nTest][nTrain];
        for (int run = 0; run < nbRuns; run++) {
            append = window != 1 || run != 0;
            System.out.print(String.format("[LB-COMPUTE-TIME] Run %d, Window size %d (%.2f)", run, window, r));

            train.randomize(new Random(run));
            test.randomize(new Random(run));

            SequenceStatsCache trainCache = new SequenceStatsCache(train);
            SequenceStatsCache testCache = new SequenceStatsCache(test);
            trainCache.setKeoghEnvelopesSorted(window);
            testCache.setKeoghEnvelopesSorted(window);

            elapsedTime = 0;
            tightness = 0;
            comparisons = 0;
            for (int queryIndex = 0; queryIndex < nTest; queryIndex++) {
                Instance query = test.instance(queryIndex);
                for (int candidateIndex = 0; candidateIndex < nTrain; candidateIndex++) {

                    Instance candidate = train.instance(candidateIndex);
                    double dtwDist = dtwComputer.distance(query, candidate);

                    switch (lowerBound) {
                        case "LbKim":
                            startTime = System.nanoTime();
                            dist = kimComputer.distance(query, candidate, testCache, trainCache, queryIndex, candidateIndex);
                            stopTime = System.nanoTime();
                            break;
                        case "LbKeogh":
                            startTime = System.nanoTime();
                            dist = keoghComputer.distance(query, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex]);
                            stopTime = System.nanoTime();
                            break;
                        case "LbImproved":
                            startTime = System.nanoTime();
                            dist = improvedComputer.distance(query, candidate, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex], window);
                            stopTime = System.nanoTime();
                            break;
                        case "LbNew":
                            startTime = System.nanoTime();
                            dist = newComputer.distance(query, candidate, trainCache.sortedSequence[candidateIndex]);
                            stopTime = System.nanoTime();
                            break;
                        case "LbEnhanced":
                            startTime = System.nanoTime();
                            dist = enhancedComputer.distance(query, candidate, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex], window, enhancedParam);
                            stopTime = System.nanoTime();
                            break;
                    }

                    if (dtwDist > 0) {
                        comparisons++;
                        elapsedTimes[queryIndex][candidateIndex] = stopTime - startTime;
                        elapsedTime += elapsedTimes[queryIndex][candidateIndex];

                        tightnesss[queryIndex][candidateIndex] = dist / dtwDist;
                        tightness += tightnesss[queryIndex][candidateIndex];
                    }
                }
            }
            elapsedTime /= comparisons;
            tightness /= comparisons;

            if (lowerBound.equals("LbEnhanced"))
                System.out.println(String.format(" -- %s%d: %3d s %10d ns -- %6.3f",
                        lowerBound, enhancedParam, (int) (elapsedTime / 1e9), (int) (elapsedTime % 1e9), tightness));
            else
                System.out.println(String.format(" -- %s: %3d s %10d ns -- %6.3f",
                        lowerBound, (int) (elapsedTime / 1e9), (int) (elapsedTime % 1e9), tightness));

            save(filename, run, window, r, tightnesss, elapsedTimes, append);
        }

        System.out.println("");
    }


    private static void save(String filename, int run, int win, double r, double[][] tightness, double[][] time, boolean append) {
        FileWriter out;
        final String comma = ",";
        final int n = tightness.length;
        final int m = tightness[0].length;

        try {
            out = new FileWriter(filename, append);
            if (!append)
                out.append("Run,r,Win,Pairs,Time,Tightness\n");
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    String str = i + "--" + j;
                    out.append(String.valueOf(run)).append(comma)
                            .append(String.valueOf(r)).append(comma)
                            .append(String.valueOf(win)).append(comma)
                            .append(str).append(comma)
                            .append(String.valueOf(time[i][j])).append(comma)
                            .append(String.valueOf(tightness[i][j])).append("\n");
                }
            }
            out.flush();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
