function Final_Mode = FMD(fs, x, FilterSize, CutNum, ModeNum, MaxIterNum)
    %
    %---------------
    % Input:
    %---------------
    %
    %   fs              :       sampling frequency of x
    %   x               :       signal to be analyzed
    %   FilterSiz       :       filter size
    %   CutNum          :       the cut number of the whole frequency band
    %   ModeNum         :       the final mode number
    %   MaxIterNum      :       max iteration number
    %
    %---------------
    % Output:
    %---------------
    %
    %   Final_Mode      :       final mode
    %
    %---------------
    % Reference:
    %---------------
    %       Y. Miao, B. Zhang, C. Li, J. Lin, D. Zhang
    %       "Feature Mode Decomposition:New Decomposition Theory
    %       for Rotating Machinery Fault Diagnosis"
    %       IEEE Transactions on Industrial Electronics.2022
    %       DOI:10.1109/TIE.2022.3156156
    %-------------------------------------------------
    %
    % Author:  Boyao Zhang AND Yonghao Miao
    % Version: 2022/02
    %
    %-------------------------------------------------

    % Initialization
    freq_bound = 0:1 / CutNum:1 - 1 / CutNum;

    temp_filters = zeros(FilterSize, CutNum);
    for n = 1:length(freq_bound)
        w = window(@hanning,FilterSize);
        temp_filters(:, n) = fir1(FilterSize - 1, [freq_bound(n) + eps, freq_bound(n) + 1 / CutNum - eps], w)';
    end

    % result initialization
    result = cell(CutNum + 1, 5);
    result(1, :) = {'IterCount', 'Iterations', 'CorrMatrix', 'ComparedModeNum', 'StopNum'};

    % Iterations
    temp_sig = repmat(x, [1, CutNum]);

    itercount = 2;
    while 1
        iternum = 2;
        if itercount == 2
            iternum = MaxIterNum - (CutNum - ModeNum) * iternum;
        end

        result{itercount, 1} = iternum;

        for n = 1:size(temp_filters, 2)

            f_init = temp_filters(:, n);
            [y_Iter, f_Iter, k_Iter, T_Iter] = xxc_mckd(fs, temp_sig(:, n), f_init, iternum, [], 1, 0);
            result{itercount, 2}{n, 1} = y_Iter(:, end);
            result{itercount, 2}{n, 2} = f_Iter(:, end);
            result{itercount, 2}{n, 3} = k_Iter(:, end);
            result{itercount, 2}{n, 4} = abs(fft(f_Iter));
            result{itercount, 2}{n, 4} = result{itercount, 2}{n, 4}(1:FilterSize / 2, :);
            [~, index1] = max(result{itercount, 2}{n, 4});
            result{itercount, 2}{n, 5} = (index1 - 1) * (fs / FilterSize);
            result{itercount, 2}{n, 6} = T_Iter;

        end

        X = result{itercount, 2};

        for n = 1:size(X, 1)
            temp_sig(:, n) = X{n, 1}(:, end);
            temp_filters(:, n) = X{n, 2}(:, end);
        end

        CorrMatrix = abs(corrcoef(temp_sig));
        CorrMatrix = triu(CorrMatrix, 1);

        [I, J, ~] = max_IJ(CorrMatrix);
        Location = [I, J];

        XI = X{I, 1}(:, end);
        XJ = X{J, 1}(:, end);
        XI = XI - mean(XI);
        XJ = XJ - mean(XJ);

        T_1 = result{itercount, 2}{I, 6};
        KI = CK(XI, T_1, 1);
        T_2 = result{itercount, 2}{J, 6};
        KJ = CK(XJ, T_2, 1);

        if KI > KJ
            output = J;
        else
            output = I;
        end

        X(output, :) = [];
        temp_sig(:, output) = [];
        temp_filters(:, output) = [];

        result{itercount, 3} = CorrMatrix;
        result{itercount, 4} = Location;
        result{itercount, 5} = output;

        if size(temp_filters, 2) == ModeNum-1
            break
        end

        itercount = itercount + 1;

    end
    %
    for nn = 1:size(result{itercount, 2}, 1)
        Final_Mode(:, nn) = result{itercount, 2}{nn, 1}(:, end);
    end
end
function [y_final f_final ckIter T_final] = xxc_mckd(fs, x, f_init, termIter, T, M, plotMode)
    %---------------
    % Reference:
    %---------------
    %   Y. Miao, M. Zhao, J. Lin, Y. Lei
    %       "Application of an improved maximum correlated kurtosis deconvolution method
    %       for fault diagnosis of rolling element bearings"
    %       Mechanical Systems and Signal Processing, 92 (2017) 173 - 195.
    %   Y. Miao, M. Zhao, K. Liang, J. Lin
    %       "Application of an improved MCKDA for fault detection
    %       of wind turbine gear based on encoder signal"
    %       Renewable Energy, 151 (2020) 192 - 203.
    %-------------------------------------------------
    %
    % Author: @ Yonghao Miao
    %
    %-------------------------------------------------

    % Assign default values for inputs

    if (isempty(termIter))
        termIter = 30;
    end
    if (isempty(plotMode))
        plotMode = 0;
    end
    if (isempty(M))
        M = 3;
    end

    %%----------------------------
    if (isempty(T))
        xxenvelope = abs(hilbert(x)) - mean(abs(hilbert(x)));
        T = TT(xxenvelope, fs);
    end

    T = round(T);

    %%----------------------------
    x = x(:);
    L = length(f_init);
    N = length(x);

    % Calculate XmT
    XmT = zeros(L, N, M + 1);
    for (m = 0:M)
        for (l = 1:L)
            if (l == 1)
                XmT(l, (m * T + 1):end, m + 1) = x(1:N - m * T);
            else
                XmT(l, 2:end, m + 1) = XmT(l - 1, 1:end - 1, m + 1);
            end
        end
    end

    % Calculate the matrix inverse section
    Xinv = inv(XmT(:, :, 1) * XmT(:, :, 1)');

    f = f_init;
    ck_best = 0;

    % Initialize iteration matrices
    y = zeros(N, 1);
    b = zeros(L, 1);
    ckIter = [];

    n = 1;

    while n == 1 || (n <= termIter)
        % Compute output signal
        y = (f' * XmT(:, :, 1))';

        % Generate yt
        f_final(:, n) = f;
        yt = zeros(N, M);
        for (m = 0:M)
            if (m == 0)
                yt(:, m + 1) = y;
            else
                yt(T + 1:end, m + 1) = yt(1:end - T, m);
            end
        end

        % Calculate alpha
        alpha = zeros(N, M + 1);
        for (m = 0:M)
            alpha(:, m + 1) = (prod(yt(:, [1:m (m + 2):size(yt, 2)]), 2).^2) .* yt(:, m + 1);
        end

        % Calculate beta
        beta = prod(yt, 2);

        % Calculate the sum Xalpha term
        Xalpha = zeros(L, 1);
        for (m = 0:M)
            Xalpha = Xalpha + XmT(:, :, m + 1) * alpha(:, m + 1);
        end

        % Calculate the new filter coefficients
        f = sum(y.^2) / (2 * sum(beta.^2)) * Xinv * Xalpha;
        f = f / sqrt(sum(f.^2)); 
        % Calculate the ck value of this iteration
        ckIter(n) = sum(prod(yt, 2).^2) / (sum(y.^2)^(M + 1));
        % Update the best match filter if required
        if (ckIter(n) > ck_best)
            ck_best = ckIter(n);
        end

        %% Update the period
        %%----------------------------
        xyenvelope = abs(hilbert(y)) - mean(abs(hilbert(y)));
        T = TT(xyenvelope, fs);
        T = round(T);
        T_final = T;

        % Calculate XmT
        XmT = zeros(L, N, M + 1);
        for (m = 0:M)
            for (l = 1:L)
                if (l == 1)
                    XmT(l, (m * T + 1):end, m + 1) = x(1:N - m * T);
                else
                    XmT(l, 2:end, m + 1) = XmT(l - 1, 1:end - 1, m + 1);
                end
            end
        end

        % Calculate the matrix inverse section
        Xinv = inv(XmT(:, :, 1) * XmT(:, :, 1)');

        %%----------------------------
        y_final(:, n) = filter(f_final(:, n), 1, x);
        n = n + 1;
    end
end
function [T] = TT (y, fs)
    % estimate the period in y based on auto-correlation function.
    %---------------
    % Input:
    %---------------
    %
    %   y       :       signal to be analyzed
    %   fs      :       sampling frequency
    %
    %---------------
    % Output:
    %---------------
    %
    %   T       :       estimated period in sample
    %
    %-------------------------------------------------

    % find the maximum lag M
    M = fs;

    NA = xcorr(y, y, M, 'coeff');
    NA = NA(ceil(length(NA) / 2):end);

    % find first zero-crossing
    sample1 = NA(1);

    for lag = 2:length(NA)
        sample2 = NA(lag);

        if ((sample1 > 0) && (sample2 < 0))
            zeroposi = lag;
            break;
        elseif ((sample1 == 0) || (sample2 == 0))
            zeroposi = lag;
            break;
        else
            sample1 = sample2;
        end

    end

    % Cut from the first zero-crossing
    NA = NA(zeroposi:end);
    % Find the max position (time lag)
    % corresponding the max aside from the first zero-crossing
    [~, max_position] = max(NA);
    % Give the extimated period by autocorrelation
    T = zeroposi + max_position;

end

function ck = CK(x, T, M)

    % Calculate the Correlated Kurtosis

    if nargin < 3
        M = 2;
    end

    % Insure x is a row vector
    x = x(:)';

    N = length(x);
    x_shift = zeros(M + 1, N);

    x_shift(1, :) = x;

    for m = 1:M
        x_shift(m + 1, T + 1:end) = x_shift(m, 1:end - T);
    end

    ck = sum(prod(x_shift).^2) / sum(x.^2)^(M + 1);

end

function [I, J, M] = max_IJ(X)
    % Returns the row and column indices of the maximum in matrix X.

    [temp, tempI] = max(X);
    [M, J] = max(temp);
    I = tempI(J);
end
