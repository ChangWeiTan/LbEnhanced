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
 along with LbEnhanced.  If not, see <http://www.gnu.org/licenses/>.*/
package utilities;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * General tools to do simple operations
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class GenericTools {
    public static double distanceTo(final double a, final double b) {
        double diff = a - b;
        return diff * diff;
    }

    public static int argMin3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
    }

    public static int argMax3(final double a, final double b, final double c) {
        return (a >= b) ? ((a >= c) ? 0 : 2) : (b >= c) ? 1 : 2;
    }

    public static int argMax4(final double a, final double b, final double c, final double d) {
        if (a >= b && a >= c && a >= d) return 0;
        else if (b >= a && b >= c && b >= d) return 1;
        else if (c >= a && c >= b && c >= d) return 2;
        else return 3;
    }

    public static double min3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? a : c) : (b <= c) ? b : c;
    }
}
