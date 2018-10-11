# LbEnhanced
This is the source code for LbEnhanced, a new lower bound for the DTW distance

## Running the code:
Running from terminal
1. Classification Time and Pruning Power (Order training data with Euclidean Distance)
* java -Xmx14g -Xms14g -cp $LIBDIR: experiments.PruneAndTimeSortEuclidean $PROJECTDIR $DATASETDIR $PROBLEM $RUNS $WIN $TOESTIMATEORNOT $LOWERBOUND $ENHANCEDPARAM

2. Tightness of the lower bounds
* java -Xmx14g -Xms14g -cp $LIBDIR: experiments.Tightness $PROJECTDIR $DATASETDIR $PROBLEM $WIN $LOWERBOUND $ENHANCEDPARAM

Running from Bash Script
1. bash runPruneTimeSortEuclidean.sh
2. bash runPruneTimeRandomSampling.sh
3. bash runTightness.sh
