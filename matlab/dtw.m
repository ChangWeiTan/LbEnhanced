function distance = dtw(seriesA, seriesB, W, matrixD)
n = length(seriesA);
m = length(seriesB);

matrixD(1,1) = (seriesA(1) - seriesB(1))^2;
for i = 2:min(n, W+1)
    matrixD(i,1) = matrixD(i-1, 1) + (seriesA(i) - seriesB(1))^2;
end
for j = 2:min(m, W+1)
    matrixD(1,j) = matrixD(1, j-1) + (seriesA(1) - seriesB(j))^2;
end
if j <= m
    matrixD(1, j) = inf;
end

for i = 2:n
    jStart = max(2, i-W);
    jEnd = min(m, i+W);
    indexLeft = i-W-1;
    if indexLeft >= 1
        matrixD(i, indexLeft) = inf;
    end
    
    for j = jStart:jEnd
        minRes = min(matrixD(i-1,j-1),min(matrixD(i-1,j),matrixD(i,j-1)));
        matrixD(i,j) = minRes + (seriesA(i)-seriesB(j))^2;
    end
    if j <= m
        matrixD(1, j) = inf;
    end
end

distance = matrixD(n,m);

end