#!/bin/bash 
PROJECTDIR=$PWD/
DATASETDIR="/home/ubuntu/workspace/Dataset/TSC_Problems/"
if [ ! -d "$DATASETDIR" ]; then
      DATASETDIR="/mnt/c/Users/cwtan/workspace/Dataset/TSC_Problems/"
fi

METHODS=("LbKim" "LbKeogh" "LbImproved" "LbNew")
LARGE=("WormsTwoClass" "Earthquakes" "WordsSynonyms" "Strawberry"
            "CricketX" "CricketY" "CricketZ" "FiftyWords" "FaceAll" "InsectWingbeatSound" "Computers"
            "ECG5000" "ChlorineConcentration" "Haptics" "TwoPatterns" "LargeKitchenAppliances"
            "RefrigerationDevices" "ScreenType" "SmallKitchenAppliances" "ShapesAll" "Mallat" "wafer"
            "CinCECGtorso" "yoga" "InlineSkate" "UWaveGestureLibraryX" "UWaveGestureLibraryY"
            "UWaveGestureLibraryZ" "Phoneme" "ElectricDevices" "FordB" "FordA" "NonInvasiveFatalECGThorax1"
            "NonInvasiveFatalECGThorax2" "HandOutlines" "UWaveGestureLibraryAll" "StarLightCurves")
SMALL=("ItalyPowerDemand" "SonyAIBORobotSurface1" "Coffee" "ECG200"
            "BeetleFly" "BirdChicken" "SonyAIBORobotSurface2" "Wine" "GunPoint" "TwoLeadECG" "MoteStrain" "Beef"
            "Plane" "FaceFour" "OliveOil" "SyntheticControl" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxTW"
            "ECGFiveDays" "MiddlePhalanxTW" "MiddlePhalanxOutlineAgeGroup" "ArrowHead" "CBF" "Lightning7"
            "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW" "ToeSegmentation2" "DiatomSizeReduction"
            "ToeSegmentation1" "Meat" "Trace" "ShapeletSim" "DistalPhalanxOutlineCorrect" "Herring"
            "MiddlePhalanxOutlineCorrect" "ProximalPhalanxOutlineCorrect" "Car" "Lightning2" "Ham" "MedicalImages"
            "Symbols" "Adiac" "SwedishLeaf" "FISH" "FacesUCR" "OSULeaf" "PhalangesOutlinesCorrect" "Worms")
PROBLEM=HandOutlines
SAMPLESIZE=500
SERIESLEN=256
WINDOWS=(1 2 3 4 5 6 7 8 9 10)
RS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
RUNS=6
echo Compiling javac -sourcepath src -d bin -cp $PWD/lib/*: src/**/*.java 
javac -sourcepath src -d bin -cp $PWD/lib/*: src/**/*.java

cd bin 
echo Current Directory: $PWD
echo Dataset Directory: $DATASETDIR

for method in ${METHODS[@]}; do 
      for win in ${WINDOWS[@]}; do
            java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.LbTime $PROJECTDIR $DATASETDIR $PROBLEM $RUNS $win $method 5 $SAMPLESIZE $SERIESLEN
      done
      for r in ${RS[@]}; do
            java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.LbTime $PROJECTDIR $DATASETDIR $PROBLEM $RUNS $r $method 5 $SAMPLESIZE $SERIESLEN
      done
done
for i in {1..5}; do
      for win in ${WINDOWS[@]}; do
            java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.LbTime $PROJECTDIR $DATASETDIR $PROBLEM $RUNS $win LbEnhanced $i $SAMPLESIZE $SERIESLEN
      done
      for r in ${RS[@]}; do
            java -Xmx14g -Xms14g -cp $PROJECTDIR/lib/*: experiments.LbTime $PROJECTDIR $DATASETDIR $PROBLEM $RUNS $r LbEnhanced $i $SAMPLESIZE $SERIESLEN
      done
done

      