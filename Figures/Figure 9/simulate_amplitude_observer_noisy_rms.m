clearvars; close all;

load_dir = "./expt_stim_exp4new_v7";

mat_group1 = ["glass","ceramic","metal","stone"];
mat_group2 = ["wood","plastic"];

num_iter =400;
% Preallocate arrays outside the loops
decision = zeros(num_iter, 140);
mod_ans = zeros(num_iter, 140);
mod_ans_cond = zeros(num_iter, 5, 7);
matcond_idx = zeros(1, 140);
mod_ans_all = zeros(num_iter, 5, 7,4);

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

        window_size = min([round(0.04 * fs), length(y1), length(y2)]);  % Window length in samples.

        y1_segment = y1(1:window_size);
        
        y2_segment = y2(1:window_size);
         volume1 = 20*log10(rms(y1));
        volume2 = 20*log10(rms(y2));

        % Add noise to volume estimates
        noise_level = 6; % Standard deviation of noise
        volume1 = volume1 + noise_level*randn;
        volume2 = volume2 + noise_level*randn;
        % Choose the segment with the larger noisy RMS level.
        if mean(volume1) > mean(volume2)
            decision(ii,jj) = 1;
        else
            decision(ii,jj) = 2;
        end
   
    end

    mod_ans(ii,:) = 1-abs(stim_key3(:,4)'-decision(ii,:));

    % Divide mod_ans into conditions based on stim_key(:,3)
    for kkk = 1:5
        for ijk = [1 2 3 4 5 6 7]
            ind = find(stim_key3(:,3)'==kkk & stim_key3(:,1)'==ijk);
            mod_ans_cond(ii,kkk,ijk) = mean(mod_ans(ii,ind));
            mod_ans_all(ii,kkk,ijk,:) = mod_ans(ii,ind);
        end
    end

end

save("model_10dBnoise_0.04_amp_rms_400_v4.mat","mod_ans_all")

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

