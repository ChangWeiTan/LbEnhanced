function distance = lbNew(seriesA, seriesB, W, Ub, Lb)
n = length(seriesA);
m = length(seriesB);

distance = (seriesA(1)-seriesB(1))^2 + (seriesA(n)-seriesB(m))^2;
for i = 2:n-1
    if seriesA(i) > Ub(i)
        distance = distance + (seriesA(i)-Ub(i))^2;
    elseif seriesA(i) < Lb(i)
        distance = distance + (seriesA(i)-Lb(i))^2;
    else
        start = max(i-W, 1);
        stop = min(i+W, n);
        minDist = inf;
        for j = start:stop
            dist = (seriesA(i)-seriesB(j))^2;
            if dist < minDist
                minDist = dist;
            end
        end
        distance = distance + minDist;
    end
end

end