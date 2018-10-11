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
 * Implementation of LbImproved
 *
 * Original implementation:
 * Lemire, D. (2009). Faster retrieval with a two-pass dynamic-time-warping lower bound. Pattern recognition, 42(9), 2169-2180.
 * https://github.com/lemire/lbimproved
 * https://arxiv.org/abs/0811.3301
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbImproved extends DTWLowerBound {
    public double distance(final Instance a, final Instance b, final double[] Ub, final double[] Lb, final int r) {
        final int length = Math.min(Ub.length, a.numAttributes() - 1);
        final double[] y = new double[length];
        final double[] Ux = new double[length];
        final double[] Lx = new double[length];

        int i;
        double res = 0;
        double diff, c;

        for (i = 0; i < length; i++) {
            c = a.value(i);
            if (c < Lb[i]) {
                diff = Lb[i] - c;
                res += diff * diff;
                y[i] = Lb[i];
            } else if (Ub[i] < c) {
                diff = c - Ub[i];
                res += diff * diff;
                y[i] = Ub[i];
            } else {
                y[i] = c;
            }
        }

        fillImprovedEnvelope(y, r, Ux, Lx);
        for (i = 0; i < length; i++) {
            c = b.value(i);
            if (c < Lx[i]) {
                diff = Lx[i] - c;
                res += diff * diff;
            } else if (Ux[i] < c) {
                diff = c - Ux[i];
                res += diff * diff;
            }
        }

        return res;
    }

    public double distance(final Instance a, final Instance b, final double[] Ub, final double[] Lb, final int r, final double cutOffValue) {
        final int length = Math.min(Ub.length, a.numAttributes() - 1);
        final double[] y = new double[length];
        int i;
        double res = 0;
        double diff, c;

        for (i = 0; i < length; i++) {
            c = a.value(i);
            if (c < Lb[i]) {
                diff = Lb[i] - c;
                res += diff * diff;
                y[i] = Lb[i];
            } else if (Ub[i] < c) {
                diff = c - Ub[i];
                res += diff * diff;
                y[i] = Ub[i];
            } else {
                y[i] = c;
            }
        }
        if (res < cutOffValue) {
            final double[] Ux = new double[length];
            final double[] Lx = new double[length];
            fillImprovedEnvelope(y, r, Ux, Lx);
            for (i = 0; i < length; i++) {
                c = b.value(i);
                if (c < Lx[i]) {
                    diff = Lx[i] - c;
                    res += diff * diff;
                } else if (Ux[i] < c) {
                    diff = c - Ux[i];
                    res += diff * diff;
                }
            }
        }
        return res;
    }

    public double distanceWithoutKeogh(final Instance a, final Instance b, final int r, final double[] y, final double keoghDistance) {
        final int length = Math.min(b.numAttributes() - 1, a.numAttributes() - 1);
        int i;
        double res = keoghDistance;
        double diff, c;

        final double[] Ux = new double[length];
        final double[] Lx = new double[length];
        fillImprovedEnvelope(y, r, Ux, Lx);
        for (i = 0; i < length; i++) {
            c = b.value(i);
            if (c < Lx[i]) {
                diff = Lx[i] - c;
                res += diff * diff;
            } else if (Ux[i] < c) {
                diff = c - Ux[i];
                res += diff * diff;
            }
        }

        return res;
    }
}
