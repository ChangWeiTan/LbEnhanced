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
import sequences.UCRArchive;
import utilities.DataLoader;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Lower bound compute time vs tightness for random sampled dataset
 * java -Xmx14g -Xms14g -cp $LIBDIR: experiments.LbTime $PROJECTDIR $DATASETDIR $PROBLEM $RUNS $WIN $LOWERBOUND $ENHANCEDPARAM $SAMPLESIZE $SERIESLEN
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbTime extends Experiments {
    private static int enhancedParam;
    private static String lowerBound;
    private static int sampleSize;
    private static int seriesLength;

    public static void main(String[] args) {
        final long heapSize = Runtime.getRuntime().totalMemory();
        final long heapMaxSize = Runtime.getRuntime().maxMemory();
        final long heapFreeSize = Runtime.getRuntime().freeMemory();
        System.out.println(String.format("[LB-COMPUTE-TIME] Heap:      %20d", heapSize));
        System.out.println(String.format("[LB-COMPUTE-TIME] Max Heap:  %20d", heapMaxSize));
        System.out.println(String.format("[LB-COMPUTE-TIME] Free Heap: %20d", heapFreeSize));

        problem = "Adiac";
        nbRuns = 5;
        enhancedParam = 1;
        lowerBound = "LbEnhanced";
        r = -1;
        window = -1;
        sampleSize = 500;
        seriesLength = 256;

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
        if (args.length >= 8) sampleSize = Integer.parseInt(args[7]);
        if (args.length >= 9) seriesLength = Integer.parseInt(args[8]);

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
        System.out.println(String.format("[LB-COMPUTE-TIME] Sample Size:    %d", sampleSize));
        System.out.println(String.format("[LB-COMPUTE-TIME] Series Length:  %d", seriesLength));
        System.out.println("");

        singleProblem();
    }

    private static void singleProblem() {
        Instances temp = DataLoader.loadTrain("ArrowHead", datasetPath);
        Instances data = new Instances(temp, sampleSize);
        final int seed = 10;
        final Random random = new Random(seed);
        final int nTrain = Math.max(10, sampleSize / UCRArchive.sortedDataset.length / 2);

        String problem;
        while (data.numInstances() < sampleSize) {
            final int select = random.nextInt(UCRArchive.sortedDataset.length);
            problem = UCRArchive.sortedDataset[select];
            System.out.println("[LB-COMPUTE-TIME] Loading " + problem + " - current size " + data.numInstances());
            Instances train = DataLoader.loadTrain(problem, datasetPath);
            Instances test = DataLoader.loadTest(problem, datasetPath);
            if (train.numAttributes() - 1 < seriesLength) continue;

            for (int i = 0; i < nTrain; i++) {
                if (data.numInstances() >= sampleSize) break;

                int randStart = random.nextInt(train.numAttributes() - seriesLength);
                int randEnd = randStart + seriesLength;
                Instance instance = new DenseInstance(seriesLength);
                for (int k = randStart; k < randEnd; k++) {
                    instance.setValue(k - randStart, train.instance(i).value(k));
                }
                data.add(instance);
            }

            for (int i = 0; i < nTrain; i++) {
                if (data.numInstances() >= sampleSize) break;
                int randStart = random.nextInt(test.numAttributes() - seriesLength);
                int randEnd = randStart + seriesLength;
                Instance instance = new DenseInstance(seriesLength);
                for (int k = randStart; k < randEnd; k++) {
                    instance.setValue(k - randStart, test.instance(i).value(k));
                }
                data.add(instance);
            }
        }

        System.out.println("[LB-COMPUTE-TIME] Total data size: " + data.numInstances());
        if (r < 0)
            r = 1.0 * window / seriesLength;
        else if (window < 0)
            window = (int) (r * seriesLength);

        long startTime = 0, stopTime = 0;
        double dist = 0;

        DTW dtwComputer = new DTW();
        dtwComputer.setWindowSize(window);

        LbKim kimComputer = new LbKim();
        LbKeogh keoghComputer = new LbKeogh();
        LbImproved improvedComputer = new LbImproved();
        LbNew newComputer = new LbNew();
        LbEnhanced enhancedComputer = new LbEnhanced();
        LbEnhanced1 enhanced1Computer = new LbEnhanced1();
        LbEnhanced2 enhanced2Computer = new LbEnhanced2();

        problem = "data_N" + sampleSize + "L" + seriesLength;
        String filename;
        setResDir("out_sdm/lb_time/" + problem + "/");
        if (lowerBound.equals("LbEnhanced"))
            filename = resDir + problem + "_" + lowerBound + enhancedParam + ".csv";
        else
            filename = resDir + problem + "_" + lowerBound + ".csv";

        double elapsedTime, tightness;
        double elapsedTime2;
        int comparisons;
        int nData = data.numInstances();
        double[][] elapsedTimes = new double[nData][nData];
        double[][] tightnesss = new double[nData][nData];

        System.out.print(String.format("[LB-COMPUTE-TIME] Window size %d (%.2f)", window, r));

        SequenceStatsCache dataCache = new SequenceStatsCache(data);
        dataCache.setKeoghEnvelopesSorted(window);

        elapsedTime = 0;
        tightness = 0;
        comparisons = 0;
        for (int queryIndex = 0; queryIndex < nData; queryIndex++) {
            Instance query = data.instance(queryIndex);
            for (int candidateIndex = 0; candidateIndex < nData; candidateIndex++) {
                if (candidateIndex == queryIndex) continue;

                Instance candidate = data.instance(candidateIndex);
                double dtwDist = dtwComputer.distance(query, candidate);

                elapsedTime2 = 0;
                for (int run = 0; run < nbRuns; run++) {
                    switch (lowerBound) {
                        case "LbKim":
                            startTime = System.nanoTime();
                            dist = kimComputer.distance(query, candidate, dataCache, dataCache, queryIndex, candidateIndex);
                            stopTime = System.nanoTime();
                            break;
                        case "LbKeogh":
                            startTime = System.nanoTime();
                            dist = keoghComputer.distance(query, dataCache.upperEnvelope[candidateIndex],
                                    dataCache.lowerEnvelope[candidateIndex]);
                            stopTime = System.nanoTime();
                            break;
                        case "LbImproved":
                            startTime = System.nanoTime();
                            dist = improvedComputer.distance(query, candidate, dataCache.upperEnvelope[candidateIndex],
                                    dataCache.lowerEnvelope[candidateIndex], window);
                            stopTime = System.nanoTime();
                            break;
                        case "LbNew":
                            startTime = System.nanoTime();
                            dist = newComputer.distance(query, candidate, dataCache.sortedSequence[candidateIndex]);
                            stopTime = System.nanoTime();
                            break;
                        case "LbEnhanced":
                            if (enhancedParam == 1) {
                                startTime = System.nanoTime();
                                dist = enhanced1Computer.distance(query, candidate,
                                        dataCache.upperEnvelope[candidateIndex], dataCache.lowerEnvelope[candidateIndex]);
                                stopTime = System.nanoTime();
                            } else if (enhancedParam == 2) {
                                startTime = System.nanoTime();
                                dist = enhanced2Computer.distance(query, candidate, dataCache.upperEnvelope[candidateIndex],
                                        dataCache.lowerEnvelope[candidateIndex], window);
                                stopTime = System.nanoTime();
                            } else {
                                startTime = System.nanoTime();
                                dist = enhancedComputer.distance(query, candidate, dataCache.upperEnvelope[candidateIndex],
                                        dataCache.lowerEnvelope[candidateIndex], window, enhancedParam);
                                stopTime = System.nanoTime();
                            }
//                            startTime = System.nanoTime();
//                            dist = enhancedComputer.distance(query, candidate, dataCache.upperEnvelope[candidateIndex],
//                                    dataCache.lowerEnvelope[candidateIndex], window, enhancedParam);
//                            stopTime = System.nanoTime();
                            break;
                    }
                    if (run > 0)
                        elapsedTime2 += stopTime - startTime;
                }
                if (dtwDist > 0) {
                    comparisons++;
                    elapsedTimes[queryIndex][candidateIndex] = elapsedTime2 / (nbRuns + 1);
                    elapsedTime += elapsedTimes[queryIndex][candidateIndex];

                    tightnesss[queryIndex][candidateIndex] = dist / dtwDist;
                    tightness += tightnesss[queryIndex][candidateIndex];
                }
            }
        }
        elapsedTime /= comparisons;
        tightness /= comparisons;

        if (lowerBound.equals("LbEnhanced"))
            System.out.println(String.format(" -- %s%d: %3d s %10d ns -- %6.6f",
                    lowerBound, enhancedParam, (int) (elapsedTime / 1e9), (int) (elapsedTime % 1e9), tightness));
        else
            System.out.println(String.format(" -- %s: %3d s %10d ns -- %6.6f",
                    lowerBound, (int) (elapsedTime / 1e9), (int) (elapsedTime % 1e9), tightness));

        boolean append = window != 1;
        save(filename, window, r, tightnesss, elapsedTimes, append);


        System.out.println("");
    }


    private static void save(String filename, int win, double r, double[][] tightness, double[][] time, boolean append) {
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
                    if (j == i) continue;

                    String str = i + "--" + j;
                    out.append(String.valueOf(r)).append(comma)
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
