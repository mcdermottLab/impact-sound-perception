clearvars; close all;

% Ideal observer model for amplitude comparison on LEVEL-NORMALIZED stimuli
% This version normalizes each sound segment to the same RMS level before
% computing the 40ms amplitude, removing the overall amplitude cue.

load_dir = "./expt_stim_exp4new_v7";

mat_group1 = ["glass","ceramic","metal","stone"];
mat_group2 = ["wood","plastic"];

num_iter = 400;

% Preallocate arrays
decision = zeros(num_iter, 140);
mod_ans = zeros(num_iter, 140);
mod_ans_cond = zeros(num_iter, 5, 7);
matcond_idx = zeros(1, 140);
mod_ans_all = zeros(num_iter, 5, 7, 4);

% Target RMS level for normalization
target_rms = 0.1;

for ii = 1:num_iter
    stim_key = readmatrix(load_dir + "/" + num2str(ii) + "/stim_key.csv");
    stim_key2 = readtable(load_dir + "/" + num2str(ii) + "/stim_key.csv");
    stim_key3 = stim_key(:,:);
    
    for jj = 1:140
        soundfile = load_dir + "/" + num2str(ii) + "/" + num2str(jj) + ".wav";
        [y, fs] = audioread(soundfile);

        material1 = string(stim_key2.Var2(jj));
        
        % Extract the two trial segments, excluding the interval between them.
        y1 = y(1:44100);
        y2 = y(44100 + 4411:end);

        % Level normalize each segment to the same RMS
        y1_rms = rms(y1);
        y2_rms = rms(y2);
        
        if y1_rms > 0
            y1 = y1 * (target_rms / y1_rms);
        end
        if y2_rms > 0
            y2 = y2 * (target_rms / y2_rms);
        end

        % Calculate volume for the first 40ms of each normalized segment
        window_size = min([round(0.04 * fs), length(y1), length(y2)]);
        
        y1_segment = y1(1:window_size);
        y2_segment = y2(1:window_size);
        
        volume1 = 20 * log10(rms(y1_segment));
        volume2 = 20 * log10(rms(y2_segment));

        % Add noise to volume estimates
        noise_level = 6; % Standard deviation of noise in dB
        volume1 = volume1 + noise_level * randn;
        volume2 = volume2 + noise_level * randn;

        % Make decision based on which segment sounds louder
        if mean(volume1) > mean(volume2)
            decision(ii, jj) = 1;
        else
            decision(ii, jj) = 2;
        end
    end

    mod_ans(ii,:) = 1 - abs(stim_key3(:,4)' - decision(ii,:));

    % Divide mod_ans into conditions based on stim_key(:,3)
    for kkk = 1:5
        for ijk = [1 2 3 4 5 6 7]
            ind = find(stim_key3(:,3)' == kkk & stim_key3(:,1)' == ijk);
            mod_ans_cond(ii, kkk, ijk) = mean(mod_ans(ii, ind));
            mod_ans_all(ii, kkk, ijk, :) = mod_ans(ii, ind);
        end
    end
end

save("model_amp40ms_level_normalized_400_v1.mat", "mod_ans_all")

fprintf('Level-normalized amplitude model complete.\n');
fprintf('Results saved to model_amp40ms_level_normalized_400_v1.mat\n');

