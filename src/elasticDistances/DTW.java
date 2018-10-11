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
package elasticDistances;

import utilities.GenericTools;
import weka.core.Instance;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Dynamic Time Warping implementation with warping windows support
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class DTW extends ElasticDistances {
    private int w;       // warping paramId in terms of sequence length
    private double r;    // warping paramId in terms of percentage

    @Override
    public double distance(final Instance first, final Instance second) {
        return distance(first, second, w);
    }

    @Override
    public double distance(final Instance first, final Instance second, final double cutOffValue) {
        return distance(first, second, w, cutOffValue);
    }

    public double distance(final Instance first, final Instance second, final int windowSize) {
        final int n = first.numAttributes() - 1;
        final int m = second.numAttributes() - 1;

        final int winPlus1 = windowSize + 1;
        double diff;
        int i, j, jStart, jEnd, indexInfyLeft;

        diff = first.value(0) - second.value(0);
        matrixD[0][0] = diff * diff;
        for (i = 1; i < Math.min(n, winPlus1); i++) {
            diff = first.value(i) - second.value(0);
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < Math.min(m, winPlus1); j++) {
            diff = first.value(0) - second.value(j);
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }
        if (j < m)
            matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + winPlus1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0)
                matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                diff = first.value(i) - second.value(j);
                matrixD[i][j] = GenericTools.min3(matrixD[i - 1][j - 1], matrixD[i][j - 1], matrixD[i - 1][j]) + diff * diff;
            }
            if (j < m)
                matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        return matrixD[n - 1][m - 1];
    }

    public double distance(final Instance first, final Instance second, final int windowSize, final double cutOffValue) {
        boolean tooBig;
        final int n = first.numAttributes() - 1;
        final int m = second.numAttributes() - 1;

        double diff;
        int i, j, jStart, jEnd, indexInfyLeft;

        diff = first.value(0) - second.value(0);
        matrixD[0][0] = diff * diff;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first.value(i) - second.value(0);
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first.value(0) - second.value(j);
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }
        if (j < m)
            matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            tooBig = true;
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + windowSize + 1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0)
                matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                diff = first.value(i) - second.value(j);
                matrixD[i][j] = GenericTools.min3(matrixD[i - 1][j - 1], matrixD[i][j - 1], matrixD[i - 1][j]) + diff * diff;
                if (tooBig && matrixD[i][j] < cutOffValue)
                    tooBig = false;
            }
            //Early abandon
            if (tooBig)
                return Double.POSITIVE_INFINITY;

            if (j < m)
                matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        return matrixD[n - 1][m - 1];
    }

    public double distance(final double[] first, final double[] second) {
        final int n = first.length;
        final int m = second.length;

        double diff;
        int i, j;

        diff = first[0] - second[0];
        matrixD[0][0] = diff * diff;
        for (i = 1; i < n; i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < m; j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }

        for (i = 1; i < n; i++) {
            for (j = 1; j < m; j++) {
                diff = first[i] - second[j];
                matrixD[i][j] = GenericTools.min3(matrixD[i - 1][j - 1], matrixD[i][j - 1], matrixD[i - 1][j]) + diff * diff;
            }
        }

        return matrixD[n - 1][m - 1];
    }

    public double distance(final double[] first, final double[] second, final int windowSize) {
        final int n = first.length;
        final int m = second.length;
        final int winPlus1 = windowSize + 1;
        double diff;
        int i, j, jStart, jEnd, indexInfyLeft;

        diff = first[0] - second[0];
        matrixD[0][0] = diff * diff;
        for (i = 1; i < Math.min(n, winPlus1); i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + diff * diff;
        }

        for (j = 1; j < Math.min(m, winPlus1); j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + diff * diff;
        }
        if (j < m)
            matrixD[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + winPlus1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0)
                matrixD[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                diff = first[i] - second[j];
                matrixD[i][j] = GenericTools.min3(matrixD[i - 1][j - 1], matrixD[i][j - 1], matrixD[i - 1][j]) + diff * diff;
            }
            if (j < m)
                matrixD[i][j] = Double.POSITIVE_INFINITY;
        }

        return matrixD[n - 1][m - 1];
    }

    public void setWindowSize(int win) {
        w = win;
    }

    public double getR() {
        return r;
    }

    public void setR(double x) {
        r = x;
    }
}
