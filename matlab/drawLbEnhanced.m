function distance = drawLbEnhanced(fig, seriesA, seriesB, W, Ub, Lb, V, D)
if nargin < 8
    D = inf;
end
n = length(seriesA);
m = length(seriesB);

nBands = min(V, floor(n/2));
figure(fig)
plot(seriesA, 'r', 'linewidth', 2);
hold on
plot(seriesB, 'b', 'linewidth', 2);
plot(Ub, '--k', 'linewidth', 2);
plot(Lb, '--k', 'linewidth', 2);
xlim([0, length(seriesA)+1]);
line([1, 1], [seriesA(1) seriesB(1)], 'color', 'm')
line([n, m], [seriesA(n) seriesB(m)], 'color', 'm')

distance = (seriesA(1) - seriesB(1))^2 + (seriesA(n) - seriesB(m))^2;
for i = 2:nBands
    ir = n-i+1;
    ixLeftA = i;
    ixLeftB = i;
    ixRightA = ir;
    ixRightB = ir;
    
    minL = (seriesA(i) - seriesB(i))^2;
    minR = (seriesA(ir) - seriesB(ir))^2;
    for j = max(1, i-W):i-1
        jr = n-j+1;
        dist = (seriesA(i)-seriesB(j))^2;
        if dist < minL
            ixLeftB = j;
            minL = dist;
        end
        dist = (seriesA(j)-seriesB(i))^2;
        if dist < minL
            ixLeftA = j;
            minL = dist;
        end
        
        dist = (seriesA(ir)-seriesB(jr))^2;
        if dist < minR
            ixRightB = jr;
            minR = dist;
        end
        dist = (seriesA(jr)-seriesB(ir))^2;
        if dist < minR
            ixRightA = jr;
            minR = dist;
        end
    end
    line([ixLeftA, ixLeftB], [seriesA(ixLeftA) seriesB(ixLeftB)], 'color', 'm')
    line([ixRightA, ixRightB], [seriesA(ixRightA) seriesB(ixRightB)], 'color', 'm')
    distance = distance + minL + minR;
end
if distance > D
    distance = inf;
else
    start = nBands+1;
    stop = n-nBands;
    for i = start:stop
        if seriesA(i) > Ub(i)
            distance = distance + (seriesA(i)-Ub(i))^2; 
            line([i, i], [seriesA(i) Ub(i)], 'color', 'g')
        end
        if seriesA(i) < Lb(i)
            distance = distance + (seriesA(i)-Lb(i))^2;
            line([i, i], [seriesA(i) Lb(i)], 'color', 'g')
        end
    end
end
hold off
legend('Series A', 'Series B', 'Envelopes for B', 'location', 'best')
title(sprintf('LbEnhanced_{%d}^{%d}(A,B)=%.3f',W,V,distance));
axis off

end