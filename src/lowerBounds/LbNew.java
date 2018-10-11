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
 * Implementation of LbNew
 *
 * Original implementation:
 * Shen, Y., Chen, Y., Keogh, E., & Jin, H. (2018, May). Accelerating Time Series Searching with Large Uniform Scaling.
 * In Proceedings of the 2018 SIAM International Conference on Data Mining (pp. 234-242). Society for Industrial and Applied Mathematics.
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbNew extends DTWLowerBound {
    public double distance(final Instance a, final Instance b, final double[][] bSorted) {
        final int length = a.numAttributes() - 1;
        final int n = length - 1;
        int i, m;
        double diff, val;

        diff = a.value(0) - b.value(0);
        double res = diff * diff;
        for (i = 1; i < n; i++) {
            m = bSorted[i].length - 1;
            val = a.value(i);
            if (val < bSorted[i][0]) {
                // min
                diff = bSorted[i][0] - val;
            } else if (val > bSorted[i][m]) {
                // max
                diff = val - bSorted[i][m];
            } else {
                diff = findMinimum(val, bSorted[i]);
            }
            res += diff * diff;
        }
        diff = a.value(n) - b.value(n);
        res += diff * diff;

        return res;
    }

    private double findMinimum(final double a, final double[] arr) {
        return binarySearch(a, arr, Double.POSITIVE_INFINITY, 0, arr.length - 1);
    }

    private double binarySearch(final double a, final double[] arr, double bsf, int left, int right) {
        if (right >= left) {
            final int mid = left + (right - left) / 2;
            double distance = a - arr[mid];
            if (distance < 0) {
                // a < arr[mid], go left
                right = mid - 1;
                if (-distance < bsf) bsf = -distance;
                return binarySearch(a, arr, bsf, left, right);
            } else if (distance > 0) {
                // a > arr[mid], go right;
                left = mid + 1;
                if (distance < bsf) bsf = distance;
                return binarySearch(a, arr, bsf, left, right);
            } else {
                return distance;
            }
        }
        return bsf;
    }


}
