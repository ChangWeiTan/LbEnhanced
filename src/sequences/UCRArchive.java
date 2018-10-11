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
package sequences;

import utilities.Path;

import java.io.File;

/**
 * Code for the paper "Elastic bands across the path: A new framework and method to lower bound DTW"
 *
 * Stores dataset names for the Standard UCR Archive
 * Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen and Gustavo Batista (2015).
 * The UCR Time Series Classification Archive.
 * URL www.cs.ucr.edu/~eamonn/time_series_data/
 *
 * @author Chang Wei Tan, Francois Petitjean, Geoff Webb
 */
public class UCRArchive {
    /**
     * Sorted in increasing DTW computations per test series
     */
    public static String[] sortedDataset = new String[]{"ItalyPowerDemand", "SonyAIBORobotSurface1", "Coffee", "ECG200",
            "BeetleFly", "BirdChicken", "SonyAIBORobotSurface2", "Wine", "GunPoint", "TwoLeadECG", "MoteStrain", "Beef",
            "Plane", "FaceFour", "OliveOil", "SyntheticControl", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW",
            "ECGFiveDays", "MiddlePhalanxTW", "MiddlePhalanxOutlineAgeGroup", "ArrowHead", "CBF", "Lightning7",
            "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW", "ToeSegmentation2", "DiatomSizeReduction",
            "ToeSegmentation1", "Meat", "Trace", "ShapeletSim", "DistalPhalanxOutlineCorrect", "Herring",
            "MiddlePhalanxOutlineCorrect", "ProximalPhalanxOutlineCorrect", "Car", "Lightning2", "Ham", "MedicalImages",
            "Symbols", "Adiac", "SwedishLeaf", "FISH", "FacesUCR", "OSULeaf", "PhalangesOutlinesCorrect", "Worms",
            "WormsTwoClass", "Earthquakes", "WordSynonyms", "Strawberry", "CricketX", "CricketY", "CricketZ",
            "FiftyWords", "FaceAll", "InsectWingbeatSound", "Computers", "ECG5000", "ChlorineConcentration", "Haptics",
            "TwoPatterns", "LargeKitchenAppliances", "RefrigerationDevices", "ScreenType", "SmallKitchenAppliances",
            "ShapesAll", "Mallat", "wafer", "CinCECGtorso", "yoga", "InlineSkate", "UWaveGestureLibraryX",
            "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Phoneme", "ElectricDevices", "FordB", "FordA",
            "NonInvasiveFatalECGThorax1", "NonInvasiveFatalECGThorax2", "HandOutlines", "UWaveGestureLibraryAll", "StarLightCurves"};

    /**
     * Datasets that are small and fast to classify
     */
    public static String[] smallDataset = new String[]{"ItalyPowerDemand", "SonyAIBORobotSurface1", "Coffee", "ECG200",
            "BeetleFly", "BirdChicken", "SonyAIBORobotSurface2", "Wine", "GunPoint", "TwoLeadECG", "MoteStrain", "Beef",
            "Plane", "FaceFour", "OliveOil", "SyntheticControl", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW",
            "ECGFiveDays", "MiddlePhalanxTW", "MiddlePhalanxOutlineAgeGroup", "ArrowHead", "CBF", "Lightning7",
            "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW", "ToeSegmentation2", "DiatomSizeReduction",
            "ToeSegmentation1", "Meat", "Trace", "ShapeletSim", "DistalPhalanxOutlineCorrect", "Herring",
            "MiddlePhalanxOutlineCorrect", "ProximalPhalanxOutlineCorrect", "Car", "Lightning2", "Ham", "MedicalImages",
            "Symbols", "Adiac", "SwedishLeaf", "FISH", "FacesUCR", "OSULeaf", "PhalangesOutlinesCorrect", "Worms"};

    public static String[] largeDataset = new String[]{"WormsTwoClass", "Earthquakes", "WordSynonyms", "Strawberry",
            "CricketX", "CricketY", "CricketZ", "FiftyWords", "FaceAll", "InsectWingbeatSound", "Computers",
            "ECG5000", "ChlorineConcentration", "Haptics", "TwoPatterns", "LargeKitchenAppliances",
            "RefrigerationDevices", "ScreenType", "SmallKitchenAppliances", "ShapesAll", "Mallat", "wafer",
            "CinCECGtorso", "yoga", "InlineSkate", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ", "Phoneme", "ElectricDevices", "FordB", "FordA", "NonInvasiveFatalECGThorax1",
            "NonInvasiveFatalECGThorax2", "HandOutlines", "UWaveGestureLibraryAll", "StarLightCurves"};

    /**
     * New datasets used in
     * Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles (COTE) and
     * Time series classification with ensembles of elastic distance measures (EE)
     * <p>
     * Refer to http://www.timeseriesclassification.com/dataset.php
     */
    public static String[] newTSCProblems = new String[]{"ElectricDeviceOn", "EpilepsyX",
            "EthanolLevel", "HeartbeatBIDMC", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2"};

    public static int totalDatasets() {
        return newTSCProblems.length + sortedDataset.length;
    }

    public static String total() {
        return newTSCProblems.length + sortedDataset.length + " in total.";
    }

    public static void main(String[] args) {
        // check if the dataset exists
        for (String aSortedDataset : sortedDataset) {
            StringBuilder problem2 = new StringBuilder(aSortedDataset);
            System.out.print("Checking " + aSortedDataset + "  ");
            String fullPath = Path.datasetPath + aSortedDataset + "/" + aSortedDataset + "_TRAIN";
            File dir = new File(fullPath);
            int count = 0;
            while (!dir.exists() && count < 2) {
                if (count == 0) {
                    // try lower case
                    problem2 = new StringBuilder(aSortedDataset.toLowerCase());
                    fullPath = Path.datasetPath + problem2 + "/" + problem2 + "_TRAIN";
                    dir = new File(fullPath);
                    count++;
                } else if (count == 1) {
                    // try removing _
                    String[] tmp = aSortedDataset.split("_");
                    problem2 = new StringBuilder(tmp[0]);
                    for (int j = 1; j < tmp.length; j++) {
                        problem2.append(tmp[j]);
                    }
                    fullPath = Path.datasetPath + problem2 + "/" + problem2 + "_TRAIN";
                    dir = new File(fullPath);
                    count++;
                }
            }
            if (count == 0) System.out.println("");
            else System.out.println(" changed to " + problem2);
        }
    }
}
