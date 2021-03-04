clear
clc
close all

rng(5)

n = 100;
a = zeros(1,n);
b = zeros(1,n);
a(1) = rand(1);
b(1) = rand(1);
for i = 2:n
    r = rand(1);
    if r > 0.5
        a(i) = a(i-1)+rand(1)/10;
    else
        a(i) = a(i-1)-rand(1)/10;
    end
    
    r = rand(1);
    if r > 0.5
        b(i) = b(i-1)+rand(1)/10;
    else
        b(i) = b(i-1)-rand(1)/10;
    end
end
t = linspace(0, 3*pi, n);
seriesA = sin(t).*a;
seriesB = sin(t+pi/2).*b;

% seriesA = [6 7 9 8 4 2 3 6 9 7 4 2];
% seriesB = [0 3 4 6 5 4 3 4 5 4 5 6];

V = 5;
W = floor(0.1*length(seriesA));
[U,L] = fillEnvelope(seriesB, W);
costM = inf*ones(n,n);

% two series
% fig=1;
% figure(fig)
% plot(seriesA, 'r', 'linewidth', 2);
% hold on
% plot(seriesB, 'b', 'linewidth', 2);
% xlim([0, length(seriesA)+1]);
% hold off
% axis off

% DTW
fig=2;
distDTW = drawDTW(fig, seriesA, seriesB, W, costM);

% LbKeogh
fig=3;
distKeogh = drawLbKeogh(fig, seriesA, seriesB, W, U, L);

% LbImproved
fig=4;
distImproved = drawLbImproved(fig, seriesA, seriesB, W, U, L);

% LbEnhanced
fig=5;
distEnhanced = drawLbEnhanced(fig, seriesA, seriesB, W, U, L, V);

% LbKim
fig=6;
distKim = drawLbKim(fig, seriesA, seriesB);

% LbYi
fig=7;
distYi = drawLbYi(fig, seriesA, seriesB);

% LbNew
fig=8;
distNew = drawLbNew(fig, seriesA, seriesB, W, U, L);
