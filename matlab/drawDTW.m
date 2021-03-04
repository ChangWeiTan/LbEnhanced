function distance = drawDTW(fig, seriesA, seriesB, W, costM)
n = length(seriesA);
m = length(seriesB);

figure(fig)
% scaledSeriesA = seriesA+max(seriesB)+abs(min(seriesA));
scaledSeriesA = seriesA;
plot(scaledSeriesA, 'r', 'linewidth', 2);
hold on
plot(seriesB, 'b', 'linewidth', 2);
xlim([0, length(seriesA)+1]);
line([1, 1], [seriesA(1) seriesB(1)], 'color', 'g')
line([n, m], [seriesA(n) seriesB(m)], 'color', 'g')

pathM = zeros(size(costM));
costM(1,1) = (seriesA(1) - seriesB(1))^2;
for i = 2:min(n, W+1)
    costM(i,1) = costM(i-1, 1) + (seriesA(i) - seriesB(1))^2;
    pathM(i,1) = 1;
end
for j = 2:min(m, W+1)
    costM(1,j) = costM(1, j-1) + (seriesA(1) - seriesB(j))^2;
    pathM(1,j) = 2;
end
if j <= m
    costM(1, j+1) = inf;
end

for i = 2:n
    jStart = max(2, i-W);
    jEnd = min(m, i+W);
    indexLeft = i-W-1;
    if indexLeft >= 1
        costM(i, indexLeft) = inf;
    end
    for j = jStart:jEnd        
        minRes = costM(i-1,j-1);
        pathM(i,j) = 0;
        if costM(i-1,j) < minRes
            minRes = costM(i-1,j);
            pathM(i,j) = 1;
        end
        if costM(i,j-1) < minRes
            minRes = costM(i,j-1);
            pathM(i,j) = 2;
        end
       
        costM(i,j) = minRes + (seriesA(i)-seriesB(j))^2;
    end
    if j <= m
        costM(1, j) = inf;
    end
end

i = n;
j = n;
while i > 1 || j > 1
    if pathM(i,j) == 0
        i=i-1;
        j=j-1;
    elseif pathM(i,j) == 1
        i=i-1;
    elseif pathM(i,j) == 2
        j=j-1;
    end
    line([i,j], [scaledSeriesA(i) seriesB(j)], 'color', 'g');
end

distance = costM(n,m);
hold off
legend('Series A', 'Series B', 'location', 'best')
title(sprintf('DTW_{%d}(A,B)=%.3f',W,distance));
axis off

end