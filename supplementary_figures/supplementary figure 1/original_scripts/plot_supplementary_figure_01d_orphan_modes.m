clearvars; close all
% Load CSV files
RT60 = readmatrix('RT60_1.csv');
gamma = readmatrix('gamma_1.csv');
freq = readmatrix('freq_1.csv');
Amp = readmatrix('Amp_1.csv');
alpha = readmatrix('alpha_1.csv');
directories = csv2cell('directories.csv','fromfile');  % Load directories.csv
recording_names = csv2cell('../recordings_full.csv', 'fromfile');
% extracted_irs = csv2cell('extracted_irs.csv', 'fromfile');  % Load extracted_irs.csv

dirnames = csv2cell('dirnames.csv', 'fromfile');  % Load dirnames.csv

% allowed_materials = ["plastic","wood","cardboard"];
allowed_materials = ["metal","glass","ceramic","stone"];
%Set figure visibility
set(0, 'DefaultFigureVisible', 'on');
% allowed_materials = ["metal","glass","ceramic"];
% allowed_materials = ["stone"];


% Extract the string between the last two "/"
extracted_names = strings(height(directories), 1);
for i = 1:height(directories)
    path = directories{i};
    parts = strsplit(path, '/');
    extracted_names(i) = string(parts{end-1});
end

extract_nums = str2double(extracted_names);

for i = 1:height(dirnames)
    path = dirnames{i};
    parts = strsplit(path, '/');
    if length(parts) >= 2
        extracted_names2(i) = string(parts{end-1});
    else
        extracted_names2(i) = string(parts{end});
    end
end

% After extracted_names2 is populated, add:
unique_objects = unique(extracted_names2);


% Filter unique_objects based on allowed materials
filtered_objects = strings(0);
for i = 1:length(unique_objects)
    if check_material(unique_objects(i), recording_names, allowed_materials)
        filtered_objects = [filtered_objects; unique_objects(i)];
    end
end
unique_objects = filtered_objects;



% Initialize containers to store results
freq_pair_counts = zeros(length(unique_objects), 1);
unpaired_modes = zeros(length(unique_objects), 1);  % New array for unpaired modes
object_names = strings(length(unique_objects), 1);
num_samples = 8;
num_samp_fin = num_samples*ones(length(unique_objects), 1);
cc = 0;
% Loop through each unique object
for obj_idx = 1:length(unique_objects)
    target_object = unique_objects(obj_idx);
    
    % Check if object's material is allowed
    if ~check_material(target_object, recording_names, allowed_materials)
        continue;
    end
    cc=cc+1;
    matching_indices = find(extracted_names2 == target_object);

    if length(matching_indices) < 8
        continue;
    end
    
    if ~isempty(matching_indices)
        % Find indices in extracted_names that match
        matching_extracted_indices = find(ismember(extracted_names, string(matching_indices)));
        
        % Instead of getting freq_values from CSV, load and process WAV files
        freq_values = zeros(length(matching_indices), 10); % Store top 10 peaks for each IR
        for ir_idx = 1:length(matching_indices)
            % Construct path to h.wav file
            wav_path = fullfile('observed', num2str(matching_indices(ir_idx)-1), 'observed_1.wav');
            
            % Load and process the audio
            [h, fs] = audioread(wav_path);
            
            % Calculate FFT
            N = length(h);
            Y = fft(h);
            P2 = abs(Y/N);
            P1 = P2(1:N/2+1);
            P1(2:end-1) = 2*P1(2:end-1);
            f = fs*(0:(N/2))/N;
            
            P1 = 20*log10(P1);
            % Smooth the spectrum
            window_size = 50;  % Adjust this value as needed
            P1_smooth = movmean(P1, window_size);
            
            % Find peaks
            [~, peak_locs] = findpeaks(P1_smooth, 'NPeaks', 10, 'SortStr', 'descend');
            peak_freqs = f(peak_locs);
            % Sort peak frequencies in ascending order
            peak_freqs = sort(peak_freqs);
            % Store the peak frequencies
            freq_values(ir_idx, 1:length(peak_freqs)) = peak_freqs;
        end

        % Randomly sample 4 indices if we have more than 4
        if size(freq_values, 1) > num_samples
            random_indices = randperm(size(freq_values, 1), num_samples);
            freq_values = freq_values(random_indices, :);
        end

        num_samp_fin(obj_idx) = min(size(freq_values, 1), num_samples);
        
        % Flatten frequency values into a single vector and remove zeros/NaNs
        freq_flat = freq_values(:);
        freq_flat = freq_flat(freq_flat > 0 & ~isnan(freq_flat));
        
        % Count pairs with relative frequency limit and track unpaired frequencies
        pair_count = 0;
        paired_freqs = false(size(freq_flat));  % Track which frequencies are paired
        for i = 1:length(freq_flat)
            for j = (i+1):length(freq_flat)
                f_lim = max(8,0.01 * freq_flat(i));  % 1% of the first frequency
                if abs(freq_flat(i) - freq_flat(j)) <= f_lim
                    pair_count = pair_count + 1;
                    paired_freqs(i) = true;
                    paired_freqs(j) = true;
                end
            end
        end
        
        % Count unpaired frequencies
        unpaired_count = sum(~paired_freqs)/num_samp_fin(obj_idx);
        
        % Store results
        freq_pair_counts(obj_idx) = pair_count;
        unpaired_modes(obj_idx) = unpaired_count;
        object_names(obj_idx) = target_object;
        
        % fprintf('Object: %s, Close frequency pairs: %d\n', target_object, pair_count);
    end
end

% Create a summary table
results_table = table(object_names, freq_pair_counts, unpaired_modes, 'VariableNames', {'Object', 'CloseFreqPairs', 'UnpairedModes'});
% disp(results_table);

% Initialize containers for random sampling results
freq_pair_counts_random = zeros(length(unique_objects), 1);
unpaired_modes_random = zeros(length(unique_objects), 1);  % New array for random unpaired modes
dd = 0;

% Create a mapping of materials to their indices
material_indices = containers.Map();
for i = 1:length(extracted_names2)
    obj_name = extracted_names2(i);
    obj_idx = find(strcmpi(recording_names(:,2), obj_name));
    if ~isempty(obj_idx)
        material = lower(string(recording_names{obj_idx(1), 11}));
        if ~material_indices.isKey(material)
            material_indices(material) = [];
        end
        matching_extracted_indices = find(ismember(extracted_names, string(i)));
        material_indices(material) = [material_indices(material); matching_extracted_indices(:)];
    end
end

% Loop through each unique object with random sampling
for obj_idx = 1:length(unique_objects)
    target_object = unique_objects(obj_idx);
    
    % Check if object's material is allowed
    if ~check_material(target_object, recording_names, allowed_materials)
        continue;
    end
    dd = dd + 1;
    
    % Get the material of the target object
    obj_idx2 = find(strcmpi(recording_names(:,2), target_object));
    material = lower(string(recording_names{obj_idx2(1), 11}));
    
    % Select random indices only from the same material
    material_valid_indices = material_indices(material);
    random_indices = material_valid_indices(randi([1, length(material_valid_indices)], 1, num_samp_fin(obj_idx)));
    freq_values = freq(random_indices, :);
    
    % Flatten frequency values into a single vector and remove zeros/NaNs
    freq_flat = freq_values(:);
    freq_flat = freq_flat(freq_flat > 0 & ~isnan(freq_flat));
    
    % Count pairs with relative frequency limit and track unpaired frequencies
    pair_count = 0;
    paired_freqs = false(size(freq_flat));  % Track which frequencies are paired
    for i = 1:length(freq_flat)
        for j = (i+1):length(freq_flat)
            f_lim = max(8,0.01 * freq_flat(i));  % 1% of the first frequency
            if abs(freq_flat(i) - freq_flat(j)) <= f_lim
                pair_count = pair_count + 1;
                paired_freqs(i) = true;
                paired_freqs(j) = true;
            end
        end
    end
    
    % Count unpaired frequencies
    unpaired_count = sum(~paired_freqs)/num_samp_fin(obj_idx);
    
    % Store results
    freq_pair_counts_random(obj_idx) = pair_count;
    unpaired_modes_random(obj_idx) = unpaired_count;
end

% Create a summary table for random sampling
results_table_random = table(object_names, freq_pair_counts_random, unpaired_modes_random, 'VariableNames', {'Object', 'CloseFreqPairsRandom', 'UnpairedModesRandom'});
% disp(results_table_random);

% Filter out zero counts before calculating statistics
nonzero_freq_pairs = freq_pair_counts(:)/8;
% nonzero_freq_pairs = freq_pair_counts(freq_pair_counts > 0);
nonzero_freq_pairs_random = freq_pair_counts_random(:)/8;
% nonzero_freq_pairs_size = freq_pair_counts_size(freq_pair_counts_size > 0);

total_freq_pairs = mean(nonzero_freq_pairs);
total_freq_pairs_random = mean(nonzero_freq_pairs_random);
std_freq_pairs = std(nonzero_freq_pairs)/sqrt(length(nonzero_freq_pairs));
std_freq_pairs_random = std(nonzero_freq_pairs_random)/sqrt(length(nonzero_freq_pairs_random));

fprintf('Total frequency pairs: %d\n', total_freq_pairs);
fprintf('Total frequency pairs (random): %d\n', total_freq_pairs_random);

% Calculate statistics for unpaired modes
mean_unpaired = mean(unpaired_modes);
mean_unpaired_random = mean(unpaired_modes_random);
std_unpaired = std(unpaired_modes)/sqrt(length(unpaired_modes));
std_unpaired_random = std(unpaired_modes_random)/sqrt(length(unpaired_modes_random));

fprintf('Average unpaired modes: %.2f ± %.2f\n', mean_unpaired, std_unpaired);
fprintf('Average unpaired modes (random): %.2f ± %.2f\n', mean_unpaired_random, std_unpaired_random);

% Add unpaired modes to the bar plot
figure('Units', 'inches', 'Position', [2 2 2 4]);
bar(1:2,[mean_unpaired, mean_unpaired_random], 'FaceColor', [0.7 0.7 0.7]);
hold on
errorbar(1:2,[mean_unpaired, mean_unpaired_random], ...
    [std_unpaired, std_unpaired_random], '.k', 'LineWidth', 2,"CapSize",0);
xticks(1:2)
xlim([0.5 2.5])
xticklabels({'Within object', 'Randomly sampled'});
ylabel('Number of unpaired modes');
title('Comparison of unpaired modes');
set(gca, 'FontSize', 12);

% Save the figure
saveas(gcf, 'unpaired_modes_comparison.svg');

% Create histogram figure
% figure('Units', 'inches', 'Position', [2 2 8 4]);

% % Create two subplots side by side
% subplot(1,2,1)
% histogram(nonzero_freq_pairs, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k');
% xlabel('Number of frequency pairs');
% ylabel('Count');
% title('Within object');
% set(gca, 'FontSize', 12);
% 
% subplot(1,2,2)
% histogram(nonzero_freq_pairs_random, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k');
% xlabel('Number of frequency pairs');
% ylabel('Count');
% title('Randomly sampled');
% set(gca, 'FontSize', 12);
% 
% % Save the histogram figure
% saveas(gcf, 'frequency_pairs_histograms.svg');

% % Add bar plot for frequency pairs
% figure('Units', 'inches', 'Position', [2 2 2.5 4]);
% bar(1:2,[total_freq_pairs, total_freq_pairs_random], 'FaceColor', [0.7 0.7 0.7]);
% hold on
% errorbar(1:2,[total_freq_pairs, total_freq_pairs_random], ...
%     [std_freq_pairs, std_freq_pairs_random], '.k', 'LineWidth', 2,"CapSize",0);
% xticks(1:2)
% xlim([0.5 2.5])
% xticklabels({'Within object', 'Randomly sampled'});
% ylabel('Number of common mode pairs');
% title('Comparison of common mode pairs');
% set(gca, 'FontSize', 12);
% 
% % Save the figure
% saveas(gcf, 'common_mode_pairs_comparison.svg');

% After unique_objects is populated, add a function to check materials
function is_allowed = check_material(obj_name, recording_names, allowed_materials)
    % Find the row where the object name appears in column 2
    obj_idx = find(strcmpi(recording_names(:,2), obj_name));
    if ~isempty(obj_idx)
        % Get the material from column 11
        material = lower(string(recording_names{obj_idx(1), 11}));
        is_allowed = any(strcmp(allowed_materials, material));
    else
        is_allowed = false;
    end
end

