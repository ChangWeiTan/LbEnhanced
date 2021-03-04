function distance = drawLbKim(fig, seriesA, seriesB)
n = length(seriesA);
m = length(seriesB);

[maxA, imaxA] = max(seriesA);
[minA, iminA] = min(seriesA);
[maxB, imaxB] = max(seriesB);
[minB, iminB] = min(seriesB);

figure(fig)
plot(seriesA, 'r', 'linewidth', 2);
hold on
plot(seriesB, 'b', 'linewidth', 2);
line([1,n], [maxA maxA], 'color', 'k', 'linestyle', '--', 'linewidth', 2)
line([1,n], [minA minA], 'color', 'k', 'linestyle', '--', 'linewidth', 2)
line([1,n], [maxB maxB], 'color', 'k', 'linestyle', '--', 'linewidth', 2)
line([1,n], [minB minB], 'color', 'k', 'linestyle', '--', 'linewidth', 2)
line([1,1], [seriesA(1),seriesB(1)], 'color', 'g');
line([n,m], [seriesA(n),seriesB(m)], 'color', 'g');
if maxA < maxB
    line([imaxA,imaxA], [maxA,maxB], 'color', 'g');
    text(imaxA*0.9, maxB*1.1, 'max')
else
    line([imaxB,imaxB], [maxA,maxB], 'color', 'g');
    text(imaxB*0.9, maxA*1.1, 'max')
end
if minA > minB
    line([iminA,iminA], [minA,minB], 'color', 'g');
    text(iminA*0.9, minB*1.1, 'min')
else
    line([iminB,iminB], [minA,minB], 'color', 'g');
    text(iminB*0.9, minA*1.1, 'min')
end
text(1.2, (seriesA(1)+seriesB(1))/2, 'first')
text(n*0.93, (seriesA(n)+seriesB(m))/2, 'last')
    
distance = (seriesA(1)-seriesB(1))^2 + (seriesA(n)-seriesB(m))^2;
if imaxA>1 && imaxB>1
    distance = distance + (maxA-maxB)^2;
end

if iminA<n && iminB>m
    distance = distance + (minA-minB)^2;
end
xlim([0, length(seriesA)+1]);
hold off
legend('Series A', 'Series B', 'location', 'best')
title(sprintf('LbKim(A,B)=%.3f',distance));
axis off

end