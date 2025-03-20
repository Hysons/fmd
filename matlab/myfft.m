function [ff, amp] = myfft(fs, x, plotMode)
    % Fourier Spectrum of x.
    %
    % Inputs:
    %
    %
    %   fs:
    %       Sampling frequency of 'x'.
    %   x:
    %       Signal for spectral analysis .
    %
    %   plotMode:
    %       plotMode == 0, No plot the spectrum. (Default)
    %       plotMode == 1, Plot the spectrum.
    %
    % Outputs:
    %
    %   ff:
    %       The frequency axis.
    %
    %   amp:
    %       Amplitude spectrum.
    %
    %

    if nargin < 3
        plotMode = 0;
    end

    NN = length(x);
    ff = linspace(0, fs, NN + 1);
    ff = ff(1:NN);
    amp = abs(fft(x) / NN * 2);

    amp = amp(1:round(NN / 2));
    ff = ff(1:round(NN / 2));

    if plotMode == 1
        plot(ff, amp, 'b');
    end

end
