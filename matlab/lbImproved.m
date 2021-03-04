function distance = lbImproved(seriesA, seriesB, W, Ub, Lb, D)
if nargin < 6
    D = inf;
end
n = length(seriesA);
m = length(seriesB);

y = zeros(size(seriesA));

distance = 0;
for i = 1:n
    if seriesA(i) > Ub(i)
        distance = distance + (seriesA(i)-Ub(i))^2;
        y(i) = Ub(i);
    elseif seriesA(i) < Lb(i)
        y(i) = Lb(i);
        distance = distance + (seriesA(i)-Lb(i))^2;
    else
        y(i) = seriesA(i);
    end
end

if distance < D
    [U,L] = fillEnvelope(y, W);
    for i = 1:n
        if seriesB(i) > U(i)
            distance = distance + (seriesB(i)-U(i))^2;
        elseif seriesB(i) < L(i)
            distance = distance + (seriesB(i)-L(i))^2;
        end
    end
end
end