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

import weka.core.Instances;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Loading arff data
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class DataLoader {
    private static String datasetPath = Path.datasetPath;

    public static Instances loadData(String fullPath) {
        if (fullPath.substring(fullPath.length() - 5, fullPath.length()).equalsIgnoreCase(".ARFF")) {
            fullPath = fullPath.substring(0, fullPath.length() - 5);
        }

        Instances d = null;
        FileReader r;
        try {
            r = new FileReader(fullPath + ".arff");
            d = new Instances(r);
            d.setClassIndex(d.numAttributes() - 1);
        } catch (IOException e) {
            System.out.println("Unable to load data on path " + fullPath + " Exception thrown =" + e);
            System.exit(0);
        }
        return d;
    }

    public static Instances loadData(File file) throws IOException {
        Instances inst = new Instances(new FileReader(file));
        inst.setClassIndex(inst.numAttributes() - 1);
        return inst;
    }

    public static Instances loadTrain(String problem) {
        String fullPath = datasetPath + problem + "/" + problem + "_TRAIN";
        return loadData(fullPath);
    }

    public static Instances loadTrain(String problem, String datasetPath) {
        String fullPath = datasetPath + problem + "/" + problem + "_TRAIN";
        return loadData(fullPath);
    }

    public static Instances loadTest(String problem) {
        String fullPath = datasetPath + problem + "/" + problem + "_TEST";
        return loadData(fullPath);
    }

    public static Instances loadTest(String problem, String datasetPath) {
        String fullPath = datasetPath + problem + "/" + problem + "_TEST";
        return loadData(fullPath);
    }
}
