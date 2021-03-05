function distance = lbEnhanced(seriesA, seriesB, Ub, Lb, W, V, D)
% Chang Wei Tan, Francois Petitjean, Geoff Webb
% 
% Elastic bands across the path: a new framework and method to lower bound DTW
% https://changweitan.com/research/elastic_band.pdf
if nargin < 7
    D = inf;
end
n = length(seriesA);
m = length(seriesB);

nBands = min(V, floor(n/2));

distance = (seriesA(1) - seriesB(1))^2 + (seriesA(n) - seriesB(m))^2;
for i = 2:nBands
    ir = n-i+1;
    minL = (seriesA(i) - seriesB(i))^2;
    minR = (seriesA(ir) - seriesB(ir))^2;
    for j = max(1, i-W):i-1
        jr = n-j+1;
        minL = min([minL, (seriesA(i)-seriesB(j))^2, (seriesA(j)-seriesB(i))^2]);
        minR = min([minR, (seriesA(ir)-seriesB(jr))^2, (seriesA(jr)-seriesB(ir))^2]);
    end
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
        end
        if seriesA(i) < Lb(i)
            distance = distance + (seriesA(i)-Lb(i))^2;
        end
    end
    
    % one liner code for LB Keogh from https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm
%     A = seriesA(start:stop);
%     U = Ub(start:stop);
%     L = Lb(start:stop);
%     lbKeogh = sum([[A > U].* [A - U]; [A < L].* [L - A]].^2);
%     distance = distance + lbKeogh;
end
end