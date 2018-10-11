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

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Tightness of each lower bound
 * java -Xmx14g -Xms14g -cp $LIBDIR: experiments.Tightness $PROJECTDIR $DATASETDIR $PROBLEM $WIN $LOWERBOUND $ENHANCEDPARAM
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class Tightness extends Experiments {
    private static int enhancedParam;
    private static String lowerBound;

    public static void main(String[] args) {
        final long heapSize = Runtime.getRuntime().totalMemory();
        final long heapMaxSize = Runtime.getRuntime().maxMemory();
        final long heapFreeSize = Runtime.getRuntime().freeMemory();
        System.out.println(String.format("[TIGHTNESS] Heap:      %20d", heapSize));
        System.out.println(String.format("[TIGHTNESS] Max Heap:  %20d", heapMaxSize));
        System.out.println(String.format("[TIGHTNESS] Free Heap: %20d", heapFreeSize));

        problem = "Adiac";
        nbRuns = 5;
        enhancedParam = 5;
        lowerBound = "LbEnhanced";
        r = -1;
        window = 2;

        if (args.length >= 1) projectPath = args[0];
        if (args.length >= 2) datasetPath = args[1];
        if (args.length >= 3) problem = args[2];
        if (args.length >= 4) {
            if (args[3].contains("."))
                r = Double.parseDouble(args[3]);
            else
                window = Integer.parseInt(args[3]);
        }
        if (args.length >= 5) lowerBound = args[4];
        if (args.length >= 6) enhancedParam = Integer.parseInt(args[5]);

        System.out.println("[TIGHTNESS] Input Arguments:");
        System.out.println(String.format("[TIGHTNESS] Project Path:   %s", projectPath));
        System.out.println(String.format("[TIGHTNESS] Dataset Path:   %s", datasetPath));
        System.out.println(String.format("[TIGHTNESS] Problem:        %s", problem));
        if (r >= 0)
            System.out.println(String.format("[TIGHTNESS] r:              %.2f", r));
        else
            System.out.println(String.format("[TIGHTNESS] window:         %d", window));
        System.out.println(String.format("[TIGHTNESS] Lower Bound:    %s", lowerBound));
        System.out.println(String.format("[TIGHTNESS] Param:          %d", enhancedParam));
        System.out.println("");

        singleProblem(problem);
    }

    private static void singleProblem(String problem) {
        System.out.println("[TIGHTNESS] Loading " + problem);
        final Instances train = DataLoader.loadTrain(problem, datasetPath);
        final Instances test = DataLoader.loadTest(problem, datasetPath);
        System.out.println(String.format("[TIGHTNESS] %s loaded", problem));

        final int seqLen = train.numAttributes() - 1;
        if (r < 0)
            r = 1.0 * window / seqLen;
        else if (window < 0)
            window = (int) (r * seqLen);

        double dist = 0;

        DTW dtwComputer = new DTW();
        dtwComputer.setWindowSize(window);

        LbKim kimComputer = new LbKim();
        LbKeogh keoghComputer = new LbKeogh();
        LbImproved improvedComputer = new LbImproved();
        LbNew newComputer = new LbNew();
        LbEnhanced enhancedComputer = new LbEnhanced();

        String filename;
        setResDir("out_sdm/tightness/" + problem + "/");
        if (lowerBound.equals("LbEnhanced"))
            filename = resDir + problem + "_" + lowerBound + enhancedParam + ".csv";
        else
            filename = resDir + problem + "_" + lowerBound + ".csv";

        double tightness;
        int comparisons;
        int nTrain = Math.min(train.numInstances(), 100);
        int nTest = Math.min(test.numInstances(), 100);
        double[][] tightnesss = new double[nTest][nTrain];
        append = window != 1;
        System.out.print(String.format("[TIGHTNESS] Window size %d (%.2f)", window, r));

        SequenceStatsCache trainCache = new SequenceStatsCache(train);
        SequenceStatsCache testCache = new SequenceStatsCache(test);
        trainCache.setKeoghEnvelopesSorted(window);
        testCache.setKeoghEnvelopesSorted(window);

        tightness = 0;
        comparisons = 0;
        for (int queryIndex = 0; queryIndex < nTest; queryIndex++) {
            Instance query = test.instance(queryIndex);
            for (int candidateIndex = 0; candidateIndex < nTrain; candidateIndex++) {

                Instance candidate = train.instance(candidateIndex);
                double dtwDist = dtwComputer.distance(query, candidate);

                switch (lowerBound) {
                    case "LbKim":
                        dist = kimComputer.distance(query, candidate, testCache, trainCache, queryIndex, candidateIndex);
                        break;
                    case "LbKeogh":
                        dist = keoghComputer.distance(query, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex]);
                        break;
                    case "LbImproved":
                        dist = improvedComputer.distance(query, candidate, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex], window);
                        break;
                    case "LbNew":
                        dist = newComputer.distance(query, candidate, trainCache.sortedSequence[candidateIndex]);
                        break;
                    case "LbEnhanced":
                        dist = enhancedComputer.distance(query, candidate, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex], window, enhancedParam);
                        break;
                }

                if (dtwDist > 0) {
                    comparisons++;
                    tightnesss[queryIndex][candidateIndex] = dist / dtwDist;
                    tightness += tightnesss[queryIndex][candidateIndex];
                }
            }
        }
        tightness /= comparisons;

        if (lowerBound.equals("LbEnhanced"))
            System.out.println(String.format(" -- %s%d: %6.3f",
                    lowerBound, enhancedParam, tightness));
        else
            System.out.println(String.format(" -- %s: %6.3f",
                    lowerBound, tightness));

        save(filename, window, r, tightnesss, append);


        System.out.println("");
    }


    private static void save(String filename, int win, double r, double[][] tightness, boolean append) {
        FileWriter out;
        final String comma = ",";
        final int n = tightness.length;
        final int m = tightness[0].length;

        try {
            out = new FileWriter(filename, append);
            if (!append)
                out.append("r,Win,Pairs,Time,Tightness\n");
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    String str = i + "--" + j;
                    out.append(String.valueOf(r)).append(comma)
                            .append(String.valueOf(win)).append(comma)
                            .append(str).append(comma)
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
