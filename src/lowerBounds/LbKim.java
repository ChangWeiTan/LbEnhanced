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

import sequences.SequenceStatsCache;
import weka.core.Instance;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Implementation of LbKim
 *
 * Original implementation:
 * Kim, S. W., Park, S., & Chu, W. W. (2001). An index-based approach for similarity search supporting time warping in
 * large sequence databases. In Data Engineering, 2001. Proceedings. 17th International Conference on (pp. 607-614). IEEE.
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class LbKim extends DTWLowerBound {
    public double distance(final Instance query, final Instance reference,
                           final SequenceStatsCache queryCache, final SequenceStatsCache referenceCache,
                           final int indexQuery, final int indexReference) {
        double diff = query.value(0) - reference.value(0);
        double minDist = diff * diff;
        diff = query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2);
        minDist += diff * diff;

        if (!queryCache.isMinFirst(indexQuery) && !referenceCache.isMinFirst(indexReference) &&
                !queryCache.isMinLast(indexQuery) && !referenceCache.isMinLast(indexReference)) {
            diff = queryCache.getMin(indexQuery) - referenceCache.getMin(indexReference);
            minDist += diff * diff;
        }
        if (!queryCache.isMaxFirst(indexQuery) && !referenceCache.isMaxFirst(indexReference) &&
                !queryCache.isMaxLast(indexQuery) && !referenceCache.isMaxLast(indexReference)) {
            diff = queryCache.getMax(indexQuery) - referenceCache.getMax(indexReference);
            minDist += diff * diff;
        }
        return minDist;
    }

    public double distance(final Instance query, final Instance reference,
                           final SequenceStatsCache cache,
                           int indexQuery, int indexReference) {
        double diff = query.value(0) - reference.value(0);
        double minDist = diff * diff;
        diff = query.value(query.numAttributes() - 2) - reference.value(reference.numAttributes() - 2);
        minDist += diff * diff;

        if (!cache.isMinFirst(indexQuery) && !cache.isMinFirst(indexReference) &&
                !cache.isMinLast(indexQuery) && !cache.isMinLast(indexReference)) {
            diff = cache.getMin(indexQuery) - cache.getMin(indexReference);
            minDist += diff * diff;
        }
        if (!cache.isMaxFirst(indexQuery) && !cache.isMaxFirst(indexReference) &&
                !cache.isMaxLast(indexQuery) && !cache.isMaxLast(indexReference)) {
            diff = cache.getMax(indexQuery) - cache.getMax(indexReference);
            minDist += diff * diff;
        }
        return minDist;
    }
}
