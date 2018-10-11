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
package experiments;

import java.io.File;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Super Class for all experiments
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
abstract class Experiments {
    private final static String osName = System.getProperty("os.name");
    private final static String username = System.getProperty("user.name");

    static int nbRuns = 1;

    static String projectPath = setProjectPath();
    static String datasetPath = setDatasetPath();
    static String resDir;

    static String problem;
    static double r;
    static int window;
    static boolean append;

    private static String setProjectPath() {
        if (osName.contains("Window")) {
            projectPath = "C:/Users/" + username + "/workspace/LbEnhanced/";
        } else {
            projectPath = "/home/" + username + "/workspace/LbEnhanced/";
        }
        return projectPath;
    }

    private static String setDatasetPath() {
        if (osName.contains("Window")) {
            datasetPath = "C:/Users/" + username + "/workspace/Dataset/TSC_Problems/";
        } else {
            datasetPath = "/home/" + username + "/workspace/Dataset/TSC_Problems/";
        }
        return datasetPath;
    }

    static boolean setResDir(String folder) {
        resDir = projectPath + folder;
        return new File(resDir).mkdirs();
    }

    static void doNothing() {
        // do nothing for 5 seconds to warm up the program
        long startTime = System.nanoTime();
        double elapsed = 1.0 * (System.nanoTime() - startTime) / 1e9;
        while (elapsed <= 5) {
            System.out.print("");
            elapsed = 1.0 * (System.nanoTime() - startTime) / 1e9;
        }
        System.out.println();
    }
}
