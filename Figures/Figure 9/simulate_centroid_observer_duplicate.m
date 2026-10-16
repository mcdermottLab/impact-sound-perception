clearvars; close all;

load_dir = "./expt_stim_exp4new_v7";

mat_group1 = ["glass","ceramic","metal","stone"];
mat_group2 = ["wood","plastic"];

% Octave noise parameter - standard deviation in octaves
% 1 octave = doubling of frequency, so 0.5 octaves is substantial noise
octave_noise_std = 0.5;

num_iter = 400;
% Preallocate arrays outside the loops
decision = zeros(num_iter, 140);
mod_ans = zeros(num_iter, 140);
mod_ans_cond = zeros(num_iter, 5, 6);
matcond_idx = zeros(1, 140);
mod_ans_all = zeros(num_iter, 5, 6,4);
centroid_pairs = zeros(num_iter, 140, 2);

for ii =1:num_iter
    stim_key = readmatrix(load_dir+"/"+num2str(ii)+"/stim_key.csv");
    stim_key2 = readtable(load_dir+"/"+num2str(ii)+"/stim_key.csv");
    stim_key3 = stim_key(:,:);
    for jj = 1:140
        soundfile = load_dir+"/"+num2str(ii)+"/"+num2str(jj)+".wav";
        [y,fs] = audioread(soundfile);

        material1 = string(stim_key2.Var2(jj));
        
        % Extract the two trial segments, excluding the interval between them.
        y1 = y(1:44100);
        y2 = y(44100+4411:end);

        window_size = min([round(0.5 * fs), length(y1), length(y2)]);  % Window length in samples.
        
        y1_segment = y1(1:window_size);
        centroid1 = spectralCentroid(y1_segment, fs, 'Window', ones([window_size 1]));
        
        y2_segment = y2(1:window_size);
        centroid2 = spectralCentroid(y2_segment, fs, 'Window', ones([window_size 1]));

        % Store true centroids as pairs for this trial
        centroid_pairs(ii,jj,1) = mean(centroid1);
        centroid_pairs(ii,jj,2) = mean(centroid2);

        % Add octave noise to centroid estimates for decision making
        % Octave noise is Gaussian noise in log2(frequency) space
        c1_true = mean(centroid1);
        c2_true = mean(centroid2);
        
        % Add noise in log2 space (octaves), then convert back to linear
        c1_noisy = 2^(log2(c1_true) + octave_noise_std * randn);
        c2_noisy = 2^(log2(c2_true) + octave_noise_std * randn);

        % Make decision based on noisy spectral centroid
        if c1_noisy > c2_noisy
            decision(ii,jj) = 2;
        else
            decision(ii,jj) = 1;
        end

    end

    mod_ans(ii,:) = 1-abs(stim_key3(:,4)'-decision(ii,:));

    % Divide mod_ans into conditions based on stim_key(:,3)
    for kkk = 1:5
        for ijk = [1 2 3 4 5 6 7]
            ind = find(stim_key3(:,3)'==kkk & stim_key3(:,1)'==ijk);
            mod_ans_all(ii,kkk,ijk,:) = mod_ans(ii,ind);
        end
    end

end

save(sprintf("same_model_centroid_0.5_400_WITH_noise_%.2f.mat", octave_noise_std), "mod_ans_all", "centroid_pairs", "octave_noise_std")

function centroid = spectralCentroidLog(x, fs, windowSize)
    % SPECTRALCENTROIDLOG Compute spectral centroid using log amplitude
    % x: input signal
    % fs: sampling frequency
    % windowSize: size of the window in samples
    
    % Ensure signal is column vector
    x = x(:);
    
    % Calculate number of windows
    numWindows = floor(length(x)/windowSize);
    centroid = zeros(numWindows, 1);
    
    % Process each window
    for i = 1:numWindows
        % Extract window
        startIdx = (i-1)*windowSize + 1;
        endIdx = startIdx + windowSize - 1;
        xWindow = x(startIdx:endIdx);
        
        % Apply Hanning window
        win = hanning(windowSize);
        xWindow = xWindow .* win;
        
        % Compute FFT
        X = fft(xWindow);
        X = X(1:floor(windowSize/2)+1);  % Keep only positive frequencies
        
        % Compute magnitude spectrum and convert to dB
        mag = abs(X);
        mag_db = 20 * log10(mag + eps);  % Add eps to avoid log(0)

        % Threshold at 0dB and -50dB
        mag_db = mag_db-max(mag_db);
        mag_db = max(mag_db, -60);
        mag_db = mag_db-min(mag_db);
        
        % Create frequency vector
        f = (0:length(X)-1)' * fs / windowSize;
        
        % Calculate centroid using log amplitude
        centroid(i) = sum(f .* mag_db) / sum(mag_db);
    end
end

function rms_vals = movingRMS(signal, window_size)
    % Calculate moving RMS with specified window size
    rms_vals = zeros(size(signal));
    for i = 1:length(signal)
        start_idx = max(1, i - window_size + 1);
        window = signal(start_idx:i);
        rms_vals(i) = sqrt(mean(window.^2));
    end
end

