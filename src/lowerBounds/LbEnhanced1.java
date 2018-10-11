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
 * Direct implementation of LbEnhanced1
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbEnhanced1 extends DTWLowerBound {
    public double distance(final Instance a, final Instance b,
                           final double[] U, final double[] L,
                           final double cutOffValue) {
        final int l = a.numAttributes() - 2;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(l) - b.value(l);

        int i;
        double tmp, aVal;

        double res = d00 * d00 + dnm * dnm;

        if (res >= cutOffValue)
            return Double.POSITIVE_INFINITY;

        for (i = 1; i < l; i++) {
            aVal = a.value(i);
            if (aVal > U[i]) {
                tmp = aVal - U[i];
                res += tmp * tmp;
            } else if (aVal < L[i]) {
                tmp = L[i] - aVal;
                res += tmp * tmp;
            }
        }

        return res;
    }

    public double distance(final Instance a, final Instance b,
                           final double[] U, final double[] L) {
        final int l = a.numAttributes() - 2;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(l) - b.value(l);

        int i;
        double tmp, aVal;

        double res = d00 * d00 + dnm * dnm;

        for (i = 1; i < l; i++) {
            aVal = a.value(i);
            if (aVal > U[i]) {
                tmp = aVal - U[i];
                res += tmp * tmp;
            } else if (aVal < L[i]) {
                tmp = L[i] - aVal;
                res += tmp * tmp;
            }
        }

        return res;
    }
}
