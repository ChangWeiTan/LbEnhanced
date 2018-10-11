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

import lowerBounds.LbKeogh;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * NN-DTW with LbKeogh
 *
 * Original implementation:
 * Keogh, E., & Ratanamahatana, C. A. (2005). Exact indexing of dynamic time warping. Knowledge and information systems, 7(3), 358-386.
 * http://www.cs.ucr.edu/~eamonn/LB_Keogh.htm
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbKeoghDTW1NN extends DTW1NN {
    private LbKeogh lbComputer = new LbKeogh();

    @Override
    public double classifyInstance(Instance query) {
        double bsfDistance = Double.MAX_VALUE;
        int[] classCounts = new int[train.numClasses()];
        int c;
        double thisDist;

        for (candidateIndex = 0; candidateIndex < train.numInstances(); candidateIndex++) {
            Instance candidate = train.instance(candidateIndex);
            thisDist = lbComputer.distance(query, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex]);
            if (thisDist < bsfDistance) {
                thisDist = distance(query, candidate);
                if (thisDist < bsfDistance) {
                    bsfDistance = thisDist;
                    classCounts = new int[train.numClasses()];
                    classCounts[(int) candidate.classValue()]++;
                } else if (thisDist == bsfDistance) {
                    classCounts[(int) candidate.classValue()]++;
                }
            } else {
                skippedDTW++;
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

    @Override
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
            thisDist = lbComputer.distance(query, trainCache.upperEnvelope[candidateIndex], trainCache.lowerEnvelope[candidateIndex]);
            if (thisDist < bsfDistance) {
                thisDist = distance(query, candidate);
                if (thisDist < bsfDistance) {
                    bsfDistance = thisDist;
                    classCounts = new int[train.numClasses()];
                    classCounts[(int) candidate.classValue()]++;
                } else if (thisDist == bsfDistance) {
                    classCounts[(int) candidate.classValue()]++;
                }
            } else {
                skippedDTW++;
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

    @Override
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
            thisDist = distances[c].value;
            Instance candidate = train.instance(candidateIndex);

            if (thisDist < bsfDistance) {
                thisDist = distance(query, candidate);
                if (thisDist < bsfDistance) {
                    bsfDistance = thisDist;
                    classCounts = new int[train.numClasses()];
                    classCounts[(int) candidate.classValue()]++;
                } else if (thisDist == bsfDistance) {
                    classCounts[(int) candidate.classValue()]++;
                }
            } else {
                skippedDTW++;
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

    @Override
    public void init(final Instances train, final Instances test, final int window) {
        this.window = window;
        this.train = train;
        this.test = test;
        this.seqLen = train.numAttributes() - 1;
        this.trainCache.initKeogh(train);
        this.testCache.initKeogh(test);
        this.trainCache.setKeoghEnvelopes(window);
        this.testCache.setKeoghEnvelopes(window);
        distComputer.setWindowSize(window);
        distances = new Tuple[train.numInstances()];
    }
}
