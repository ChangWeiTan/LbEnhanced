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

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * A class to store LbKeogh and its cumulative distances
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbKeoghDistance {
    public double distance;
    public double distanceToNBands;
    public double[] y;

    LbKeoghDistance(final double distance, final double distanceToNBands) {
        this.distance = distance;
        this.distanceToNBands = distanceToNBands;
    }

    LbKeoghDistance(final double distance, final double[] y) {
        this.distance = distance;
        this.y = y;
    }
}
