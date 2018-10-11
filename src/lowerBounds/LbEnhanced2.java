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

import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Direct implementation of LbEnhanced2
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbEnhanced2 extends DTWLowerBound {
    public double distance(final Instance a, final Instance b,
                           final double[] U, final double[] L,
                           final int w, final double cutOffValue) {
        final int l = a.numAttributes() - 2;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(l) - b.value(l);

        int i, j, rightEnd, rightStart;
        double minL, minR, tmp, aVal;

        double res = d00 * d00 + dnm * dnm;
        rightEnd = l - 1;
        minL = a.value(1) - b.value(1);
        minL *= minL;
        minR = a.value(rightEnd) - b.value(rightEnd);
        minR *= minR;
        for (j = max(0, 1 - w); j < 1; j++) {
            rightStart = l - j;
            tmp = a.value(1) - b.value(j);
            minL = min(minL, tmp * tmp);
            tmp = a.value(j) - b.value(1);
            minL = min(minL, tmp * tmp);

            tmp = a.value(rightEnd) - b.value(rightStart);
            minR = min(minR, tmp * tmp);
            tmp = a.value(rightStart) - b.value(rightEnd);
            minR = min(minR, tmp * tmp);
        }
        res += minL + minR;

        if (res >= cutOffValue)
            return Double.POSITIVE_INFINITY;

        for (i = 2; i < rightEnd; i++) {
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
                           final double[] U, final double[] L,
                           final int w) {
        final int l = a.numAttributes() - 2;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(l) - b.value(l);

        int i, rightEnd;
        double minL, minR, tmp, aVal;

        double res = d00 * d00 + dnm * dnm;
        rightEnd = l - 1;
        minL = a.value(1) - b.value(1);
        minL *= minL;
        tmp = a.value(1) - b.value(0);
        minL = min(minL, tmp * tmp);
        tmp = a.value(0) - b.value(1);
        minL = min(minL, tmp * tmp);

        minR = a.value(rightEnd) - b.value(rightEnd);
        minR *= minR;
        tmp = a.value(rightEnd) - b.value(l);
        minR = min(minR, tmp * tmp);
        tmp = a.value(l) - b.value(rightEnd);
        minR = min(minR, tmp * tmp);

        res += minL + minR;

        for (i = 2; i < rightEnd; i++) {
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
