function distance = drawLbYi(fig, seriesA, seriesB)
n = length(seriesA);
m = length(seriesB);

figure(fig)
plot(seriesA, 'r', 'linewidth', 2);
hold on
plot(seriesB, 'b', 'linewidth', 2);

maxA = max(seriesA);
minA = min(seriesA);
maxB = max(seriesB);
minB = min(seriesB);
distance = 0;
if maxA-minA > maxB-minB
    maxVals = maxB*ones(size(seriesA));
    minVals = minB*ones(size(seriesA));
    for i = 1:n
        if seriesA(i) > maxB
            distance = distance + (seriesA(i)-maxB)^2;
            line([i i], [seriesA(i) maxB], 'color', 'g')
        elseif seriesA(i) < minB
            distance = distance + (seriesA(i)-minB)^2;
            line([i i], [seriesA(i) minB], 'color', 'g')
        end
    end
else
    maxVals = maxA*ones(size(seriesA));
    minVals = minA*ones(size(seriesA));
    for i = 1:n
        if seriesB(i) > maxA
            distance = distance + (seriesB(i)-maxA)^2;            
            line([i i], [seriesB(i) maxA], 'color', 'g')
        elseif seriesB(i) < minA
            distance = distance + (seriesB(i)-minA)^2;
            line([i i], [seriesB(i) minA], 'color', 'g')
        end
    end
end
plot(maxVals, 'k--', 'linewidth', 2);
plot(minVals, 'k--', 'linewidth', 2);
xlim([0, length(seriesA)+1]);
hold off
legend('Series A', 'Series B', 'location', 'best')
title(sprintf('LbYi(A,B)=%.3f',distance));
axis off

end