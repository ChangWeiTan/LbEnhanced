function distance = drawLbNew(fig, seriesA, seriesB, W, Ub, Lb)
if nargin < 8
    D = inf;
end
n = length(seriesA);
m = length(seriesB);

figure(fig)
plot(seriesA, 'r', 'linewidth', 2);
hold on
plot(seriesB, 'b', 'linewidth', 2);
plot(Ub, '--k', 'linewidth', 2);
plot(Lb, '--k', 'linewidth', 2);
xlim([0, length(seriesA)+1]);
line([1, 1], [seriesA(1) seriesB(1)], 'color', 'g')
line([n, m], [seriesA(n) seriesB(m)], 'color', 'g')

distance = (seriesA(1)-seriesB(1))^2 + (seriesA(n)-seriesB(m))^2;
for i = 2:n-1
    if seriesA(i) > Ub(i)
        distance = distance + (seriesA(i)-Ub(i))^2;
        line([i, i], [seriesA(i) Ub(i)], 'color', 'g')
    elseif seriesA(i) < Lb(i)
        distance = distance + (seriesA(i)-Lb(i))^2;
        line([i, i], [seriesA(i) Lb(i)], 'color', 'g')
    else
        start = max(i-W, 1);
        stop = min(i+W, n);
        minDist = inf;
        iB = -1;
        for j = start:stop
            dist = (seriesA(i)-seriesB(j))^2;
            if dist < minDist
                minDist = dist;
                iB = j;
            end
        end
        distance = distance + minDist;
        line([i,iB], [seriesA(i) seriesB(iB)], 'color', 'm')
    end
end
hold off
legend('Series A', 'Series B', 'Envelopes for B', 'location', 'best')
title(sprintf('LbNew_{%d}(A,B)=%.3f',W,distance));
axis off

end