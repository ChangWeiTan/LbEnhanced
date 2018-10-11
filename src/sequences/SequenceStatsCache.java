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
package sequences;

import lowerBounds.DTWLowerBound;
import weka.core.Instances;

import java.util.Arrays;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Cache to store stats for the sequences
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class SequenceStatsCache {
    public double[][] upperEnvelope;
    public double[][] lowerEnvelope;
    public double[][] upperRearEnvelope;
    public double[][] lowerRearEnvelope;
    public double[][] upperFrontEnvelope;
    public double[][] lowerFrontEnvelope;

    public double[][][] sortedSequence;
    public double[][] first3sorted;
    public double[][] last3sorted;
    public int[] indexMaxs, indexMins;

    private int seqLen, nSeq;
    private double[] mins, maxs;
    private boolean[] isMinFirst, isMinLast, isMaxFirst, isMaxLast;
    private Instances dataset;
    private IndexedDouble[][] indicesSortedByAbsoluteValue;

    public SequenceStatsCache() {

    }

    public SequenceStatsCache(Instances dataset) {
        initAll(dataset);
    }

    public void initKim(Instances dataset) {
        nSeq = dataset.numInstances();
        seqLen = dataset.instance(0).numAttributes() - 1;

        this.dataset = dataset;
        this.mins = new double[nSeq];
        this.maxs = new double[nSeq];
        this.indexMins = new int[nSeq];
        this.indexMaxs = new int[nSeq];
        this.isMinFirst = new boolean[nSeq];
        this.isMinLast = new boolean[nSeq];
        this.isMaxFirst = new boolean[nSeq];
        this.isMaxLast = new boolean[nSeq];

        double min, max, elt;
        int indexMin, indexMax, i, j;
        for (i = 0; i < dataset.numInstances(); i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;

            indexMin = -1;
            indexMax = -1;
            for (j = 0; j < dataset.instance(i).numAttributes() - 1; j++) {
                elt = dataset.instance(i).value(j);
                if (elt > max) {
                    max = elt;
                    indexMax = j;
                }
                if (elt < min) {
                    min = elt;
                    indexMin = j;
                }
            }
            indexMaxs[i] = indexMax;
            indexMins[i] = indexMin;
            mins[i] = min;
            maxs[i] = max;
            isMinFirst[i] = (indexMin == 0);
            isMinLast[i] = (indexMin == (dataset.instance(i).numAttributes() - 2));
            isMaxFirst[i] = (indexMax == 0);
            isMaxLast[i] = (indexMax == (dataset.instance(i).numAttributes() - 2));
        }
    }

    public void initKeogh(Instances dataset) {
        this.nSeq = dataset.numInstances();
        this.seqLen = dataset.instance(0).numAttributes() - 1;

        this.upperEnvelope = new double[nSeq][seqLen];
        this.lowerEnvelope = new double[nSeq][seqLen];
        this.dataset = dataset;
    }

    public void initEnhanced(Instances dataset) {
        this.nSeq = dataset.numInstances();
        this.seqLen = dataset.instance(0).numAttributes() - 1;

        this.upperEnvelope = new double[nSeq][seqLen];
        this.lowerEnvelope = new double[nSeq][seqLen];
        this.upperRearEnvelope = new double[nSeq][seqLen];
        this.lowerRearEnvelope = new double[nSeq][seqLen];
        this.upperFrontEnvelope = new double[nSeq][seqLen];
        this.lowerFrontEnvelope = new double[nSeq][seqLen];
        this.dataset = dataset;
    }

    public void initNew(Instances dataset) {
        nSeq = dataset.numInstances();
        seqLen = dataset.instance(0).numAttributes() - 1;

        this.upperEnvelope = new double[nSeq][seqLen];
        this.lowerEnvelope = new double[nSeq][seqLen];

        sortedSequence = new double[nSeq][seqLen][];

        this.dataset = dataset;
        this.indicesSortedByAbsoluteValue = new IndexedDouble[nSeq][seqLen];

        double elt;
        int i, j;
        for (i = 0; i < dataset.numInstances(); i++) {
            for (j = 0; j < dataset.instance(i).numAttributes() - 1; j++) {
                elt = dataset.instance(i).value(j);
                indicesSortedByAbsoluteValue[i][j] = new IndexedDouble(j, Math.abs(elt));
            }
            Arrays.sort(indicesSortedByAbsoluteValue[i], (v1, v2) -> -Double.compare(v1.value, v2.value));
        }
    }

    public void initAll(Instances dataset) {
        nSeq = dataset.numInstances();
        seqLen = dataset.instance(0).numAttributes() - 1;

        upperEnvelope = new double[nSeq][seqLen];
        lowerEnvelope = new double[nSeq][seqLen];
        upperRearEnvelope = new double[nSeq][seqLen];
        lowerRearEnvelope = new double[nSeq][seqLen];
        upperFrontEnvelope = new double[nSeq][seqLen];
        lowerFrontEnvelope = new double[nSeq][seqLen];

        sortedSequence = new double[nSeq][seqLen][];
        first3sorted = new double[nSeq][3];
        last3sorted = new double[nSeq][3];

        this.dataset = dataset;
        this.mins = new double[nSeq];
        this.maxs = new double[nSeq];
        this.indexMins = new int[nSeq];
        this.indexMaxs = new int[nSeq];
        this.isMinFirst = new boolean[nSeq];
        this.isMinLast = new boolean[nSeq];
        this.isMaxFirst = new boolean[nSeq];
        this.isMaxLast = new boolean[nSeq];
        this.indicesSortedByAbsoluteValue = new IndexedDouble[nSeq][seqLen];

        for (int i = 0; i < dataset.numInstances(); i++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            int indexMin = -1, indexMax = -1;
            for (int j = 0; j < dataset.instance(i).numAttributes() - 1; j++) {
                double elt = dataset.instance(i).value(j);
                if (elt > max) {
                    max = elt;
                    indexMax = j;
                }
                if (elt < min) {
                    min = elt;
                    indexMin = j;
                }
                indicesSortedByAbsoluteValue[i][j] = new IndexedDouble(j, Math.abs(elt));
            }
            indexMaxs[i] = indexMax;
            indexMins[i] = indexMin;
            mins[i] = min;
            maxs[i] = max;
            isMinFirst[i] = (indexMin == 0);
            isMinLast[i] = (indexMin == (dataset.instance(i).numAttributes() - 2));
            isMaxFirst[i] = (indexMax == 0);
            isMaxLast[i] = (indexMax == (dataset.instance(i).numAttributes() - 2));
            Arrays.sort(indicesSortedByAbsoluteValue[i], (v1, v2) -> -Double.compare(v1.value, v2.value));
        }
    }

    public void setKeoghEnvelopes(int window) {
        for (int i = 0; i < nSeq; i++) {
            DTWLowerBound.fillUpperLower(dataset.instance(i), window, upperEnvelope[i], lowerEnvelope[i]);
        }
    }

    public void setKeoghEnvelopesSorted(int window) {
        for (int i = 0; i < nSeq; i++) {
            DTWLowerBound.fillUpperLower(dataset.instance(i), window, upperEnvelope[i], lowerEnvelope[i], sortedSequence[i]);
        }
    }

    public void setEnhancedEnvelopes(int window) {
        for (int i = 0; i < nSeq; i++) {
            DTWLowerBound.fillAllEnvelopes(dataset.instance(i), window,
                    upperEnvelope[i], lowerEnvelope[i],
                    upperRearEnvelope[i], lowerRearEnvelope[i],
                    upperFrontEnvelope[i], lowerFrontEnvelope[i]);
        }
    }

    public void setNewEnvelopes(int window) {
        for (int i = 0; i < nSeq; i++) {
            DTWLowerBound.fillNewEnvelope(dataset.instance(i), window, sortedSequence[i], upperEnvelope[i], lowerEnvelope[i]);
        }
    }

    public void setAllEnvelopes(int window) {
        for (int i = 0; i < nSeq; i++) {
            DTWLowerBound.fillAllEnvelopes(dataset.instance(i), window,
                    upperEnvelope[i], lowerEnvelope[i],
                    upperRearEnvelope[i], lowerRearEnvelope[i],
                    upperFrontEnvelope[i], lowerFrontEnvelope[i],
                    sortedSequence[i], first3sorted[i], last3sorted[i]);
        }
    }

    public boolean isMinFirst(int i) {
        return isMinFirst[i];
    }

    public boolean isMaxFirst(int i) {
        return isMaxFirst[i];
    }

    public boolean isMinLast(int i) {
        return isMinLast[i];
    }

    public boolean isMaxLast(int i) {
        return isMaxLast[i];
    }

    public double getMin(int i) {
        return mins[i];
    }

    public double getMax(int i) {
        return maxs[i];
    }

    public int getIMax(int i) {
        return indexMaxs[i];
    }

    public int getIMin(int i) {
        return indexMins[i];
    }

    public int getIndexNthHighestVal(int i, int n) {
        return indicesSortedByAbsoluteValue[i][n].index;
    }

    public final double getUpperEnvelope(int i, int j) {
        return upperEnvelope[i][j];
    }

    public final double getUpperFrontEnvelope(int i, int j) {
        return upperFrontEnvelope[i][j];
    }

    public final double getUpperRearEnvelope(int i, int j) {
        return upperRearEnvelope[i][j];
    }

    public final double getLowerEnvelope(int i, int j) {
        return lowerEnvelope[i][j];
    }

    public final double getLowerFrontEnvelope(int i, int j) {
        return lowerFrontEnvelope[i][j];
    }

    public final double getLowerRearEnvelope(int i, int j) {
        return lowerRearEnvelope[i][j];
    }

    class IndexedDouble {
        double value;
        int index;

        IndexedDouble(int index, double value) {
            this.value = value;
            this.index = index;
        }
    }
}
