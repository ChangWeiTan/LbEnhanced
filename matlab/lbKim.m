function distance = lbKim(seriesA, seriesB)
n = length(seriesA);
m = length(seriesB);

[maxA, imaxA] = max(seriesA);
[minA, iminA] = min(seriesA);
[maxB, imaxB] = max(seriesB);
[minB, iminB] = min(seriesB);

distance = (seriesA(1)-seriesB(1))^2 + (seriesA(n)-seriesB(m))^2;
if imaxA>1 && imaxB>1
    distance = distance + (maxA-maxB)^2;
end

if iminA<n && iminB>m
    distance = distance + (minA-minB)^2;
end

end