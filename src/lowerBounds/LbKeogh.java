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
package lowerBounds;

import weka.core.Instance;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Implementation of LbKeogh
 *
 * Original implementation:
 * Keogh, E., & Ratanamahatana, C. A. (2005). Exact indexing of dynamic time warping. Knowledge and information systems, 7(3), 358-386.
 * http://www.cs.ucr.edu/~eamonn/LB_Keogh.htm
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbKeogh extends DTWLowerBound {
    public double distance(final Instance a, final double[] U, final double[] L) {
        final int length = a.numAttributes() - 1;
        int i;
        double c, diff;
        double res = 0;

        for (i = 0; i < length; i++) {
            c = a.value(i);
            if (U[i] < c) {
                diff = c - U[i];
                res += diff * diff;
            } else if (c < L[i]) {
                diff = L[i] - c;
                res += diff * diff;
            }
        }
        return res;
    }

    public LbKeoghDistance distancePreEnhanced(final Instance a, final double[] U, final double[] L, final int nBands) {
        final int length = a.numAttributes() - 1;
        final int lastIndex = length - nBands - 1;
        int i;
        double res = 0, res2 = 0;
        double c, diff;

        for (i = 0; i < nBands; i++) {
            c = a.value(i);
            if (U[i] < c) {
                diff = c - U[i];
                res += diff * diff;
            } else if (c < L[i]) {
                diff = L[i] - c;
                res += diff * diff;
            }
        }

        for (i = nBands; i <= lastIndex; i++) {
            c = a.value(i);
            if (U[i] < c) {
                diff = c - U[i];
                res2 += diff * diff;
            } else if (c < L[i]) {
                diff = L[i] - c;
                res2 += diff * diff;
            }
        }

        res += res2;
        for (i = lastIndex + 1; i < length; i++) {
            c = a.value(i);
            if (U[i] < c) {
                diff = c - U[i];
                res += diff * diff;
            } else if (c < L[i]) {
                diff = L[i] - c;
                res += diff * diff;
            }
        }

        return new LbKeoghDistance(res, res2);
    }

    public LbKeoghDistance distancePreImproved(final Instance a, final double[] U, final double[] L) {
        final int length = Math.min(U.length, a.numAttributes() - 1);
        final double[] y = new double[length];
        int i;
        double res = 0;
        double c, diff;

        for (i = 0; i < length; i++) {
            c = a.value(i);
            if (U[i] < c) {
                diff = c - U[i];
                res += diff * diff;
                y[i] = U[i];
            } else if (c < L[i]) {
                diff = L[i] - c;
                res += diff * diff;
                y[i] = L[i];
            } else {
                y[i] = c;
            }
        }

        return new LbKeoghDistance(res, y);
    }

    public double distance(final Instance a, final double[] U, final double[] L, final double cutOffValue) {
        final int length = Math.min(U.length, a.numAttributes() - 1);
        int i;
        double res = 0;
        double c, diff;

        for (i = 0; i < length; i++) {
            c = a.value(i);
            if (U[i] < c) {
                diff = c - U[i];
                res += diff * diff;
                if (res >= cutOffValue)
                    return Double.MAX_VALUE;
            } else if (c < L[i]) {
                diff = L[i] - c;
                res += diff * diff;
                if (res >= cutOffValue)
                    return Double.MAX_VALUE;
            }
        }

        return res;
    }

    public double distance(final double[] a, final double[] U, final double[] L) {
        final int length = Math.min(U.length, a.length);
        int i;
        double res = 0;
        double c, diff;

        for (i = 0; i < length; i++) {
            c = a[i];
            if (U[i] < c) {
                diff = c - U[i];
                res += diff * diff;
            } else if (c < L[i]) {
                diff = L[i] - c;
                res += diff * diff;
            }
        }
        return res;
    }
}
