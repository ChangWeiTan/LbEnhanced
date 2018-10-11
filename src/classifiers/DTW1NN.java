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

import elasticDistances.DTW;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Super class for NN-DTW
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class DTW1NN extends OneNearestNeighbour {
    protected int window;
    int v;
    DTW distComputer = new DTW();

    @Override
    public double distance(final Instance first, final Instance second, final double cutOffValue) {
        return distComputer.distance(first, second, cutOffValue);
    }

    @Override
    public double distance(final Instance first, final Instance second) {
        return distComputer.distance(first, second);
    }

    public void init(final Instances train, final Instances test, final int window) {
        this.window = window;
        this.train = train;
        this.test = test;
        this.seqLen = train.numAttributes() - 1;
        this.trainCache.initAll(train);
        this.testCache.initAll(test);
        this.trainCache.setAllEnvelopes(window);
        this.testCache.setAllEnvelopes(window);
        distComputer.setWindowSize(window);
        distances = new Tuple[train.numInstances()];
    }

    public void setV(final int v) {
        this.v = v;
    }
}
