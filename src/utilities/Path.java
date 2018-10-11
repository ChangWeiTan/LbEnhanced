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
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class Path {
    private final static String osName = System.getProperty("os.name");
    private final static String userName = System.getProperty("user.name");

    public static String datasetPath = setDatasetPath();

    private static String setDatasetPath() {
        if (osName.contains("Window")) {
            datasetPath = "C:/Users/" + userName + "/workspace/Dataset/TSC_Problems/";
        } else {
            datasetPath = "/home/" + userName + "/workspace/Dataset/TSC_Problems/";
        }

        return datasetPath;
    }

}
