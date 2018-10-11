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
package classifiers;

import elasticDistances.EuclideanDistance;
import lowerBounds.LbKeogh;
import sequences.SequenceStatsCache;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Super class for 1-NN for time series
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public abstract class OneNearestNeighbour extends AbstractClassifier {
    protected Instances train;
    protected Instances test;

    SequenceStatsCache trainCache = new SequenceStatsCache();
    SequenceStatsCache testCache = new SequenceStatsCache();
    EuclideanDistance edComputer = new EuclideanDistance();
    LbKeogh lbkeoghComputer = new LbKeogh();
    Tuple[] distances;

    private final double overFlowTime = 3.6e12;
    private final double overFlowIndex = 10;

    public int queryIndex;
    int candidateIndex;
    public long startTime, stopTime;
    public boolean OVERTIME;

    int seqLen;

    int skippedDTW;
    private double pruned;

    OneNearestNeighbour() {
    }

    public abstract double distance(Instance first, Instance second, double cutOffValue);

    public abstract double distance(Instance first, Instance second);

    @Override
    public void buildClassifier(Instances instances) {
        train = instances;
    }

    public double accuracy() {
        double predictClass;
        int correct = 0;
        skippedDTW = 0;

        for (queryIndex = 0; queryIndex < test.numInstances(); queryIndex++) {
            Instance query = test.instance(queryIndex);
            predictClass = classifyInstance(query);
            if (predictClass == query.classValue()) {
                correct++;
            }
        }
        pruned = 1.0 * skippedDTW / (test.numInstances() * train.numInstances());
        return 1.0 * correct / test.numInstances();
    }

    public double accuracySortEuclidean() {
        double predictClass;
        int correct = 0;
        skippedDTW = 0;

        for (queryIndex = 0; queryIndex < test.numInstances(); queryIndex++) {
            Instance query = test.instance(queryIndex);
            predictClass = classifyInstanceSortEuclidean(query);
            if (predictClass == query.classValue()) {
                correct++;
            }
        }
        pruned = 1.0 * skippedDTW / (test.numInstances() * train.numInstances());
        return 1.0 * correct / test.numInstances();
    }

    public double accuracySortLbKeogh() {
        double predictClass;
        int correct = 0;
        skippedDTW = 0;

        for (queryIndex = 0; queryIndex < test.numInstances(); queryIndex++) {
            Instance query = test.instance(queryIndex);
            predictClass = classifyInstanceSortLbKeogh(query);
            if (predictClass == query.classValue()) {
                correct++;
            }
        }
        pruned = 1.0 * skippedDTW / (test.numInstances() * train.numInstances());
        return 1.0 * correct / test.numInstances();
    }

    public double accuracyEstimate() {
        OVERTIME = false;
        skippedDTW = 0;
        double predictClass;
        int correct = 0;

        for (queryIndex = 0; queryIndex < test.numInstances(); queryIndex++) {
            Instance query = test.instance(queryIndex);
            predictClass = classifyInstance(query);
            if (predictClass == query.classValue()) {
                correct++;
            }
            stopTime = System.nanoTime();
            if ((stopTime - startTime) >= overFlowTime && queryIndex >= overFlowIndex) {
                OVERTIME = true;
                break;
            }
        }

        pruned = 1.0 * skippedDTW / (test.numInstances() * train.numInstances());
        return 1.0 * correct / test.numInstances();
    }

    public double accuracyEstimateSortEuclidean() {
        OVERTIME = false;
        skippedDTW = 0;
        double predictClass;
        int correct = 0;

        for (queryIndex = 0; queryIndex < test.numInstances(); queryIndex++) {
            Instance query = test.instance(queryIndex);
            predictClass = classifyInstanceSortEuclidean(query);
            if (predictClass == query.classValue()) {
                correct++;
            }
            stopTime = System.nanoTime();
            if ((stopTime - startTime) >= overFlowTime && queryIndex >= overFlowIndex) {
                OVERTIME = true;
                break;
            }
        }

        pruned = 1.0 * skippedDTW / (test.numInstances() * train.numInstances());
        return 1.0 * correct / test.numInstances();
    }

    public double accuracyEstimateSortLbKeogh() {
        OVERTIME = false;
        skippedDTW = 0;
        double predictClass;
        int correct = 0;

        for (queryIndex = 0; queryIndex < test.numInstances(); queryIndex++) {
            Instance query = test.instance(queryIndex);
            predictClass = classifyInstanceSortLbKeogh(query);
            if (predictClass == query.classValue()) {
                correct++;
            }
            stopTime = System.nanoTime();
            if ((stopTime - startTime) >= overFlowTime && queryIndex >= overFlowIndex) {
                OVERTIME = true;
                break;
            }
        }

        pruned = 1.0 * skippedDTW / (test.numInstances() * train.numInstances());
        return 1.0 * correct / test.numInstances();
    }

    @Override
    public double classifyInstance(Instance query) {
        double bsfDistance = Double.MAX_VALUE;
        int[] classCounts = new int[train.numClasses()];
        int c;
        double thisDist;

        for (candidateIndex = 0; candidateIndex < train.numInstances(); candidateIndex++) {
            Instance candidate = train.instance(candidateIndex);
            thisDist = distance(query, candidate);
            if (thisDist < bsfDistance) {
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                classCounts[(int) candidate.classValue()]++;
            } else if (thisDist == bsfDistance) {
                classCounts[(int) candidate.classValue()]++;
            }
        }

        double bsfClass = -1;
        double bsfCount = -1;
        for (c = 0; c < classCounts.length; c++) {
            if (classCounts[c] > bsfCount) {
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }

        return bsfClass;
    }

    public double classifyInstanceSortEuclidean(Instance query) {
        double bsfDistance = Double.MAX_VALUE;
        int[] classCounts = new int[train.numClasses()];
        int c;
        double thisDist;

        for (candidateIndex = 0; candidateIndex < train.numInstances(); candidateIndex++) {
            Instance candidate = train.instance(candidateIndex);
            thisDist = edComputer.distance(query, candidate);
            distances[candidateIndex] = new Tuple(candidateIndex, thisDist);
        }
        Arrays.sort(distances);

        for (c = 0; c < train.numInstances(); c++) {
            candidateIndex = distances[c].index;
            Instance candidate = train.instance(candidateIndex);
            thisDist = distance(query, candidate);
            if (thisDist < bsfDistance) {
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                classCounts[(int) candidate.classValue()]++;
            } else if (thisDist == bsfDistance) {
                classCounts[(int) candidate.classValue()]++;
            }
        }

        double bsfClass = -1;
        double bsfCount = -1;
        for (c = 0; c < classCounts.length; c++) {
            if (classCounts[c] > bsfCount) {
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }

        return bsfClass;
    }

    public double classifyInstanceSortLbKeogh(Instance query) {
        double bsfDistance = Double.MAX_VALUE;
        int[] classCounts = new int[train.numClasses()];
        int c;
        double thisDist;

        for (candidateIndex = 0; candidateIndex < train.numInstances(); candidateIndex++) {
            thisDist = lbkeoghComputer.distance(query, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex]);
            distances[candidateIndex] = new Tuple(candidateIndex, thisDist);
        }
        Arrays.sort(distances);

        for (c = 0; c < train.numInstances(); c++) {
            if (distances[c].value >= bsfDistance) break;

            candidateIndex = distances[c].index;

            Instance candidate = train.instance(candidateIndex);
            thisDist = distance(query, candidate);
            if (thisDist < bsfDistance) {
                bsfDistance = thisDist;
                classCounts = new int[train.numClasses()];
                classCounts[(int) candidate.classValue()]++;
            } else if (thisDist == bsfDistance) {
                classCounts[(int) candidate.classValue()]++;
            }
        }

        double bsfClass = -1;
        double bsfCount = -1;
        for (c = 0; c < classCounts.length; c++) {
            if (classCounts[c] > bsfCount) {
                bsfCount = classCounts[c];
                bsfClass = c;
            }
        }

        return bsfClass;
    }

    public double getPruned() {
        return pruned;
    }

    public class Tuple implements Comparable<Tuple> {
        int index;
        double value;
        double value2;
        double[] y;

        Tuple(int i, double v) {
            index = i;
            value = v;
        }

        Tuple(int i, double v, double v2) {
            index = i;
            value = v;
            value2 = v2;
        }

        Tuple(int i, double v, double[] y) {
            this.index = i;
            this.value = v;
            this.y = y;
        }


        @Override
        public int compareTo(Tuple o) {
            return Double.compare(this.value, o.value);
        }

    }
}
