function distance = drawLbKeogh(fig, seriesA, seriesB, W, Ub, Lb)
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

distance = 0;
for i = 1:n
    if seriesA(i) > Ub(i)
        distance = distance + (seriesA(i)-Ub(i))^2;
        line([i, i], [seriesA(i) Ub(i)], 'color', 'g')
    elseif seriesA(i) < Lb(i)
        distance = distance + (seriesA(i)-Lb(i))^2;
        line([i, i], [seriesA(i) Lb(i)], 'color', 'g')
    end
end
hold off
legend('Series A', 'Series B', 'Envelopes for B', 'location', 'best')
title(sprintf('LbKeogh_{%d}(A,B)=%.3f',W,distance));
axis off

end