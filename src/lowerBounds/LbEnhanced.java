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
 * <p>
 * Implementation of LbEnhanced
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbEnhanced extends DTWLowerBound {
    public double distance(final double[] a, final double[] b,
                           final double[] U, final double[] L,
                           final int w, final int v, final double cutOffValue) {
        final int n = a.length;
        final int m = b.length;
        final int l = n - 1;
        final int nBands = min(l / 2, v);
        final int lastIndex = l - nBands;
        final double d00 = a[0] - b[0];
        final double dnm = a[n - 1] - b[m - 1];

        int i, j, rightEnd, rightStart;
        double minL, minR, tmp, aVal;

        double res = d00 * d00 + dnm * dnm;

        for (i = 1; i < nBands; i++) {
            rightEnd = l - i;
            minL = a[i] - b[i];
            minL *= minL;
            minR = a[rightEnd] - b[rightEnd];
            minR *= minR;
            for (j = max(0, i - w); j < i; j++) {
                rightStart = l - j;
                tmp = a[i] - b[j];
                minL = min(minL, tmp * tmp);
                tmp = a[j] - b[i];
                minL = min(minL, tmp * tmp);

                tmp = a[rightEnd] - b[rightStart];
                minR = min(minR, tmp * tmp);
                tmp = a[rightStart] - b[rightEnd];
                minR = min(minR, tmp * tmp);
            }
            res += minL + minR;
        }
        if (res >= cutOffValue)
            return Double.POSITIVE_INFINITY;

        for (i = nBands; i <= lastIndex; i++) {
            aVal = a[i];
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
                           final int w, final int v, final double cutOffValue) {
        final int n = a.numAttributes() - 1;
        final int m = b.numAttributes() - 1;
        final int l = n - 1;
        final int nBands = min(l / 2, v);
        final int lastIndex = l - nBands;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(n - 1) - b.value(m - 1);

        int i, j, rightEnd, rightStart;
        double minL, minR, tmp, aVal;

        double res = d00 * d00 + dnm * dnm;

        for (i = 1; i < nBands; i++) {
            rightEnd = l - i;
            minL = a.value(i) - b.value(i);
            minL *= minL;
            minR = a.value(rightEnd) - b.value(rightEnd);
            minR *= minR;
            for (j = max(0, i - w); j < i; j++) {
                rightStart = l - j;
                tmp = a.value(i) - b.value(j);
                minL = min(minL, tmp * tmp);
                tmp = a.value(j) - b.value(i);
                minL = min(minL, tmp * tmp);

                tmp = a.value(rightEnd) - b.value(rightStart);
                minR = min(minR, tmp * tmp);
                tmp = a.value(rightStart) - b.value(rightEnd);
                minR = min(minR, tmp * tmp);
            }
            res += minL + minR;
        }
        if (res >= cutOffValue)
            return Double.POSITIVE_INFINITY;

        for (i = nBands; i <= lastIndex; i++) {
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

    public double distanceWithoutKeogh(final Instance a, final Instance b, final int w, final int nBands, final double keoghDistance) {
        final int n = a.numAttributes() - 1;
        final int m = b.numAttributes() - 1;
        final int l = n - 1;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(n - 1) - b.value(m - 1);

        int i, j, rightEnd, rightStart;
        double minL, minR, tmp;

        double res = d00 * d00 + dnm * dnm + keoghDistance;

        for (i = 1; i < nBands; i++) {
            rightEnd = l - i;
            minL = a.value(i) - b.value(i);
            minL *= minL;
            minR = a.value(rightEnd) - b.value(rightEnd);
            minR *= minR;
            for (j = max(0, i - w); j < i; j++) {
                rightStart = l - j;
                tmp = a.value(i) - b.value(j);
                minL = min(minL, tmp * tmp);
                tmp = a.value(j) - b.value(i);
                minL = min(minL, tmp * tmp);

                tmp = a.value(rightEnd) - b.value(rightStart);
                minR = min(minR, tmp * tmp);
                tmp = a.value(rightStart) - b.value(rightEnd);
                minR = min(minR, tmp * tmp);
            }
            res += minL + minR;
        }

        return res;
    }

    public double distance(final Instance a, final Instance b,
                           final double[] U, final double[] L,
                           final int w, final int v) {
        final int n = a.numAttributes() - 1;
        final int m = b.numAttributes() - 1;
        final int l = n - 1;
        final int nBands = min(l / 2, v);
        final int lastIndex = l - nBands;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(n - 1) - b.value(m - 1);

        int i, j, rightEnd, rightStart;
        double minL, minR, tmp, aVal;

        double res = d00 * d00 + dnm * dnm;

        for (i = 1; i < nBands; i++) {
            rightEnd = l - i;
            minL = a.value(i) - b.value(i);
            minL *= minL;
            minR = a.value(rightEnd) - b.value(rightEnd);
            minR *= minR;
            for (j = max(0, i - w); j < i; j++) {
                rightStart = l - j;
                tmp = a.value(i) - b.value(j);
                minL = min(minL, tmp * tmp);
                tmp = a.value(j) - b.value(i);
                minL = min(minL, tmp * tmp);

                tmp = a.value(rightEnd) - b.value(rightStart);
                minR = min(minR, tmp * tmp);
                tmp = a.value(rightStart) - b.value(rightEnd);
                minR = min(minR, tmp * tmp);
            }
            res += minL + minR;
        }

        for (i = nBands; i <= lastIndex; i++) {
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
