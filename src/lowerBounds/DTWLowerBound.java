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

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Super Class to DTW lower bounds
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public abstract class DTWLowerBound {
    public static void fillUpperLower(final Instance sequence, final int r, final double[] U, final double[] L) {
        final int length = sequence.numAttributes() - 1;
        int i, j, startR, stopR;
        double value, min, max;

        for (i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            startR = Math.max(0, i - r);
            stopR = Math.min(length - 1, i + r);

            for (j = startR; j <= stopR; j++) {
                value = sequence.value(j);
                if (value < min) min = value;
                if (value > max) max = value;
            }
            L[i] = min;
            U[i] = max;
        }
    }

    public static void fillUpperLower(final Instance sequence, final int r, final double[] U, final double[] L, final double[][] sortedSet) {
        final int length = sequence.numAttributes() - 1;
        int i, j, startR, stopR;
        double value, min, max;

        for (i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            startR = Math.max(0, i - r);
            stopR = Math.min(length - 1, i + r);
            sortedSet[i] = new double[stopR - startR + 1];
            for (j = startR; j <= stopR; j++) {
                value = sequence.value(j);
                if (value < min) min = value;
                if (value > max) max = value;
                sortedSet[i][j - startR] = value;
            }
            Arrays.sort(sortedSet[i]);
            L[i] = min;
            U[i] = max;
        }
    }

    public static void fillAllEnvelopes(final Instance sequence, final int r, final double[] U, final double[] L,
                                        final double[] Urear, final double[] Lrear, final double[] Ufront, final double[] Lfront) {
        final int length = sequence.numAttributes() - 1;
        int i, j, startR, stopR;
        double value, min, max, minRear, maxRear, minFront, maxFront;

        for (i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            minRear = Double.POSITIVE_INFINITY;
            maxRear = Double.NEGATIVE_INFINITY;
            minFront = Double.POSITIVE_INFINITY;
            maxFront = Double.NEGATIVE_INFINITY;
            startR = Math.max(0, i - r);
            stopR = Math.min(length - 1, i + r);

            for (j = startR; j <= stopR; j++) {
                value = sequence.value(j);
                min = Math.min(min, value);
                max = Math.max(max, value);
                if (j <= i) {
                    minRear = Math.min(minRear, value);
                    maxRear = Math.max(maxRear, value);
                }
                if (j >= i) {
                    minFront = Math.min(minFront, value);
                    maxFront = Math.max(maxFront, value);
                }
            }
            L[i] = min;
            U[i] = max;
            Lrear[i] = minRear;
            Urear[i] = maxRear;
            Lfront[i] = minFront;
            Ufront[i] = maxFront;
        }
    }

    public static void fillAllEnvelopes(final Instance sequence, final int r, final double[] U, final double[] L,
                                        final double[] Urear, final double[] Lrear, final double[] Ufront, final double[] Lfront,
                                        final double[][] sortedSet, final double[] first3, final double[] last3) {
        final int length = sequence.numAttributes() - 1;
        int i, j, startR, stopR;
        double value, min, max, minRear, maxRear, minFront, maxFront;

        for (i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            minRear = Double.POSITIVE_INFINITY;
            maxRear = Double.NEGATIVE_INFINITY;
            minFront = Double.POSITIVE_INFINITY;
            maxFront = Double.NEGATIVE_INFINITY;
            startR = Math.max(0, i - r);
            stopR = Math.min(length - 1, i + r);
            sortedSet[i] = new double[stopR - startR + 1];

            if (i < 3) first3[i] = sequence.value(i);
            else if (length - i <= 3) last3[length - i - 1] = sequence.value(i);

            for (j = startR; j <= stopR; j++) {
                value = sequence.value(j);
                sortedSet[i][j - startR] = value;
                min = Math.min(min, value);
                max = Math.max(max, value);
                if (j <= i) {
                    minRear = Math.min(minRear, value);
                    maxRear = Math.max(maxRear, value);
                }
                if (j >= i) {
                    minFront = Math.min(minFront, value);
                    maxFront = Math.max(maxFront, value);
                }
            }
            Arrays.sort(sortedSet[i]);
            L[i] = min;
            U[i] = max;
            Lrear[i] = minRear;
            Urear[i] = maxRear;
            Lfront[i] = minFront;
            Ufront[i] = maxFront;
        }
        Arrays.sort(first3);
        Arrays.sort(last3);
    }

    public static void fillNewEnvelope(final Instance sequence, final int r, final double[][] sortedSet, final double[] U, final double[] L) {
        final int length = sequence.numAttributes() - 1;
        int i, j, startR, stopR;
        double value;

        for (i = 0; i < length; i++) {
            startR = Math.max(0, i - r);
            stopR = Math.min(length - 1, i + r);
            sortedSet[i] = new double[stopR - startR + 1];
            for (j = startR; j <= stopR; j++) {
                value = sequence.value(j);
                sortedSet[i][j - startR] = value;
            }
            Arrays.sort(sortedSet[i]);
            U[i] = sortedSet[i][sortedSet[i].length-1];
            L[i] = sortedSet[i][0];
        }
    }


    public static void fillEnvelope(final double[] sequence, final int r, final double[] U, final double[] L) {
        final int length = sequence.length;
        int i, j, startR, stopR;
        double value, min, max;

        for (i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            startR = Math.max(0, i - r);
            stopR = Math.min(length - 1, i + r);
            for (j = startR; j <= stopR; j++) {
                value = sequence[j];
                if (value < min) min = value;
                if (value > max) max = value;
            }
            L[i] = min;
            U[i] = max;
        }
    }

    static void fillImprovedEnvelope(final double[] y, final int r, final double[] U, final double[] L) {
        Deque<Integer> u = new ArrayDeque<>();
        Deque<Integer> l = new ArrayDeque<>();
        u.addLast(0);
        l.addLast(0);
        int width = 1 + 2 * r;
        int i, index;
        for (i = 1; i < y.length; ++i) {
            if (i >= r + 1) {
                U[i - r - 1] = y[u.getFirst()];
                L[i - r - 1] = y[l.getFirst()];
            }
            if (y[i] > y[i - 1]) {
                u.removeLast();
                while (u.size() > 0) {
                    if (y[i] <= y[u.getLast()]) break;
                    u.removeLast();
                }
            } else {
                l.removeLast();
                while (l.size() > 0) {
                    if (y[i] >= y[l.getLast()]) break;
                    l.removeLast();
                }
            }
            u.addLast(i);
            l.addLast(i);
            if (i == width + u.getFirst()) {
                u.removeFirst();
            } else if (i == width + l.getFirst()) {
                l.removeFirst();
            }
        }

        for (i = y.length; i <= y.length + r; ++i) {
            index = Math.max(i - r - 1, 0);
            U[index] = y[u.getFirst()];
            L[index] = y[l.getFirst()];
            if (i - u.getFirst() >= width) {
                u.removeFirst();
            }
            if (i - l.getFirst() >= width) {
                l.removeFirst();
            }
        }
    }


    public static void fillEnvelope(final Instance sequence, final int r, final double[] U, final double[] L) {
        final int length = sequence.numAttributes() - 1;
        int i, j, startR, stopR;
        double min, max, value;

        for (i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            startR = Math.max(0, i - r);
            stopR = Math.min(length - 1, i + r);
            for (j = startR; j <= stopR; j++) {
                value = sequence.value(j);
                if (value < min) min = value;
                if (value > max) max = value;
            }
            L[i] = min;
            U[i] = max;
        }
    }

    public static void fillRearEnvelope(final Instance sequence, final int r, final double[] U, final double[] L) {
        final int length = sequence.numAttributes() - 1;
        int i, j, startR;
        double min, max, value;

        for (i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            startR = Math.max(0, i - r);
            for (j = startR; j <= i; j++) {
                value = sequence.value(j);
                if (value < min) min = value;
                if (value > max) max = value;
            }
            L[i] = min;
            U[i] = max;
        }
    }

    public static void fillFrontEnvelope(final Instance sequence, final int r, final double[] U, final double[] L) {
        final int length = sequence.numAttributes() - 1;
        int i, j, stopR;
        double min, max, value;

        for (i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            stopR = Math.min(length - 1, i + r);
            for (j = i; j <= stopR; j++) {
                value = sequence.value(j);
                if (value < min) min = value;
                if (value > max) max = value;
            }
            L[i] = min;
            U[i] = max;
        }
    }

    public static double[] findMinMax(final double[] sequence, final double[] minmax) {
        final int length = sequence.length;
        int i;
        double value;
        minmax[0] = Double.POSITIVE_INFINITY;
        minmax[1] = Double.NEGATIVE_INFINITY;
        for (i = 0; i < length; i++) {
            value = sequence[i];
            if (value < minmax[0]) minmax[0] = value;
            if (value > minmax[1]) minmax[1] = value;
        }

        return minmax;
    }
}
