%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script illustrates the use of the FMD code "FMD.m"
%
%%%%%%%%%%%%%%%%
% Reference:
%%%%%%%%%%%%%%%%
%
%       Y. Miao, B. Zhang, C. Li, J. Lin, D. Zhang
%       "Feature Mode Decomposition:New Decomposition Theory
%       for Rotating Machinery Fault Diagnosis"
%       IEEE Transactions on Industrial Electronics.2022
%       DOI:10.1109/TIE.2022.3156156
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Code by Boyao Zhang
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%%
load x
fs = 2e4;
t = (0:length(x) - 1)' / fs;

%%
figure('Name', 'Time waveform of mixed signal')
plot(t, x, 'b');
xlabel('Time [s]');
ylabel('Amplitude');

figure('Name', 'FFT amplitude spectrum of mixed signal')
[~, ~] = myfft(fs, x, 1);
xlabel('Frequency [Hz]');
ylabel('Amplitude');

%% Parameters
filtersize = 30;
cutnum = 7;
modenum = 2;
maxiternum = 20;

% FMD
tic
y_final = FMD(fs, x, filtersize, cutnum, modenum, maxiternum);
toc
% Plot
b = size(y_final, 2);

figure('Name', 'Time waveform of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    plot(t, y_final(:, k), 'b')
    xlabel('Time [s]');
    ylabel('Amplitude');
end

figure('Name', 'FFT amplitude spectrum of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    [ff, amp] = myfft(fs, y_final(:, k), 0);
    plot(ff, amp / max(amp), 'r');
    xlabel('Frequency [Hz]');
    ylabel('Amplitude');
end

figure('Name', 'Hilbert envelope spectrum of mode(s)')
for k = 1:b
    subplot(b, 1, k)
    envelope = abs(hilbert(y_final(:, k))) - mean(abs(hilbert(y_final(:, k))));
    [ff, amp] = myfft(fs, envelope, 1);
    xlabel('Frequency [Hz]');
    ylabel('Amplitude');
    xlim([0 300])
end
