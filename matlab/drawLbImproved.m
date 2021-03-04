function distance = drawLbImproved(fig, seriesA, seriesB, W, Ub, Lb, D)
if nargin < 8
    D = inf;
end
n = length(seriesA);
m = length(seriesB);

y = zeros(size(seriesA));

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
        y(i) = Ub(i);
        line([i, i], [seriesA(i) Ub(i)], 'color', 'g')
    elseif seriesA(i) < Lb(i)
        y(i) = Lb(i);
        distance = distance + (seriesA(i)-Lb(i))^2;
        line([i, i], [seriesA(i) Lb(i)], 'color', 'g')
    else
        y(i) = seriesA(i);
    end
end

if distance < D
    [U,L] = fillEnvelope(y, W);
%     plot(y, 'c', 'linewidth', 2);
%     plot(U, '--k', 'linewidth', 2);
%     plot(L, '--k', 'linewidth', 2);
    for i = 1:n
        if seriesB(i) > U(i)
            distance = distance + (seriesB(i)-U(i))^2;
            line([i, i], [seriesB(i) U(i)], 'color', 'm')
        elseif seriesB(i) < L(i)
            distance = distance + (seriesB(i)-L(i))^2;
            line([i, i], [seriesB(i) L(i)], 'color', 'm')
        end
    end
end
hold off
legend('Series A', 'Series B', 'Envelopes for B', 'location', 'best')
title(sprintf('LbImproved_{%d}(A,B)=%.3f',W,distance));
axis off

end