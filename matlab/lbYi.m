function distance = lbYi(seriesA, seriesB)
n = length(seriesA);

maxA = max(seriesA);
minA = min(seriesA);
maxB = max(seriesB);
minB = min(seriesB);
distance = 0;
if maxA-minA > maxB-minB
    for i = 1:n
        if seriesA(i) > maxB
            distance = distance + (seriesA(i)-maxB)^2;
        elseif seriesA(i) < minB
            distance = distance + (seriesA(i)-minB)^2;
        end
    end
else
    for i = 1:n
        if seriesB(i) > maxA
            distance = distance + (seriesB(i)-maxA)^2;       
        elseif seriesB(i) < minA
            distance = distance + (seriesB(i)-minA)^2;
        end
    end
end

end