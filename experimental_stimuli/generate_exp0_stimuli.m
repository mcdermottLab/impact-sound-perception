clearvars; close all

%% Experiment 0: Three-Condition Stimulus Generator
% This script generates stimuli for Experiment 0, which compares:
% Condition 0: Real recordings from all_impacts_combined
% Condition 1: Synthesized sounds using mode sampling (synthesize_ir_from_fitted_modes)
% Condition 2: Full model synthesis using statistics (synthesize_ir_from_gaussian_statistics)

% Add paths for synthesis functions
addpath("../synthesis_model");
addpath("../extra_scripts");

% Parameters
% Path to mode statistics (adjust if needed)
load_dir = "../data/modes_synth";
recordings_dir = "../data/recorded_impacts";
% Output will be saved relative to current directory
output_base_dir = "resynth_threecond";
materials = ["wood", "glass", "plastic", "metal", "stone", "ceramic", "cardboard"];
size_buckets = [2, 3];  % Only size buckets 2 and 3
num_sets = 50;
num_iterations = 4;  % 4 iterations per material/size/condition
fs = 44100;

% Load stiffness values (same as generate_exp1_stimuli.m)
stiffness_vals = readmatrix("./stiffness_values.csv")*1e7;

% Output directory (saved in current working directory)
output_base = output_base_dir;
% Randomized experimental stimuli directory
exp_stimuli_dir = "experimental_stimuli_randomized";

%% Generate all sets
for set_idx = 1:num_sets
    fprintf('Generating set %d of %d...\n', set_idx, num_sets);
    
    % Storage for all trials in this set (for randomization later)
    trial_data = {};  % Will store {audio, material, size, condition, iteration, source_file}
    trial_idx = 0;
    
    for mat_idx = 1:length(materials)
        material = materials(mat_idx);
        
        % Determine material resonance parameter
        if material == "metal" || material == "ceramic" || material == "stone" || material == "glass"
            mat_res = 1;
        else
            mat_res = 0;
        end
        
        for size_idx = 1:length(size_buckets)
            size_bucket = size_buckets(size_idx);
            size_bucket_str = string(size_bucket);
            
            %% Condition 0: Real Recordings
            % Get all recordings for this material from all_impacts_combined
            material_rec_dir = fullfile(recordings_dir, material);
            all_wav_files = get_all_wav_files(material_rec_dir);
            
            if isempty(all_wav_files)
                warning('No recordings found for material: %s', material);
                continue;
            end
            
            % Randomly select recordings (with replacement if needed)
            num_available = length(all_wav_files);
            selected_indices = randi(num_available, 1, num_iterations);
            
            % Save recordings for condition 0
            savedir_rec = output_base + "/set_" + string(set_idx) + "/" + material + "/" + size_bucket_str + "_0/";
            if ~exist(savedir_rec, 'dir')
                mkdir(savedir_rec);
            end
            
            for ii = 1:num_iterations
                [audio_data, ~] = audioread(all_wav_files{selected_indices(ii)});
                % Normalize audio
                audio_normalized = 0.8 * audio_data / max(abs(audio_data));
                audiowrite(savedir_rec + num2str(ii) + ".wav", audio_normalized, fs);
                
                % Store for randomized version
                trial_idx = trial_idx + 1;
                trial_data{trial_idx}.audio = audio_normalized;
                trial_data{trial_idx}.material = char(material);
                trial_data{trial_idx}.size_bucket = size_bucket;
                trial_data{trial_idx}.condition = 0;  % Recording
                trial_data{trial_idx}.iteration = ii;
                trial_data{trial_idx}.source_file = all_wav_files{selected_indices(ii)};
            end
            
            %% Condition 1: Synthesized sounds (mode sampling)
            % Load mode data
            mode_file = load_dir + "/" + material + "_" + size_bucket_str + "_modes.mat";
            load(mode_file);
            
            % Synthesis parameters
            mass = 0.1;
            stiff = stiffness_vals(mat_idx, size_bucket);  % Use material/size-specific stiffness
            vel = 4;
            
            % Save directory for condition 1
            savedir_synth = output_base + "/set_" + string(set_idx) + "/" + material + "/" + size_bucket_str + "_1/";
            if ~exist(savedir_synth, 'dir')
                mkdir(savedir_synth);
            end
            
            for ii = 1:num_iterations
                % Generate IR using mode sampling method
                [IR_fin, ~] = synthesize_ir_from_fitted_modes(modes, modes_noise, mat_res);
                
                % Generate impact sound (last param 0 to match Exp1)
                y_fin = generate_impact_sound(IR_fin, fs, mass, stiff, vel, 5, 0);
                
                % Retry if NaN (same as Exp1)
                while isnan(max(abs(y_fin)))
                    [IR_fin, ~] = synthesize_ir_from_fitted_modes(modes, modes_noise, mat_res);
                    y_fin = generate_impact_sound(IR_fin, fs, mass, stiff, vel, 5, 0);
                end
                
                % Normalize and save
                y_normalized = 0.8 * y_fin / max(abs(y_fin));
                audiowrite(savedir_synth + num2str(ii) + ".wav", y_normalized, fs);
                
                % Store for randomized version
                trial_idx = trial_idx + 1;
                trial_data{trial_idx}.audio = y_normalized;
                trial_data{trial_idx}.material = char(material);
                trial_data{trial_idx}.size_bucket = size_bucket;
                trial_data{trial_idx}.condition = 1;  % Mode sampling
                trial_data{trial_idx}.iteration = ii;
                trial_data{trial_idx}.source_file = 'mode_sampling';
            end
            
            %% Condition 2: Full Model synthesis (using statistics)
            % Calculate statistics from modes (as in Exp2 cond==1)
            M_c = nancov(modes');
            M_m = nanmean(modes, 2);
            n_mode_noise = size(modes_noise, 2) / 2;
            modes_noise_reshaped = reshape(modes_noise, [size(modes_noise, 1) size(modes_noise, 2) * size(modes_noise, 3)]);
            M_m_noise = squeeze(nanmean(modes_noise_reshaped, 1));
            M_c_noise = nancov(modes_noise_reshaped);
            M_c_noise = diag(diag(M_c_noise));
            
            num_modes = 10;
            
            % Save directory for condition 2
            savedir_full = output_base + "/set_" + string(set_idx) + "/" + material + "/" + size_bucket_str + "_2/";
            if ~exist(savedir_full, 'dir')
                mkdir(savedir_full);
            end
            
            for ii = 1:num_iterations
                % Generate IR using full statistical model
                [IR_fin, ~] = synthesize_ir_from_gaussian_statistics(M_m, M_c, M_m_noise, M_c_noise, num_modes, mat_res);
                
                % Generate impact sound
                y_fin = generate_impact_sound(IR_fin, fs, mass, stiff, vel, 5, 0);
                
                % Retry if NaN
                while isnan(max(abs(y_fin)))
                    [IR_fin, ~] = synthesize_ir_from_gaussian_statistics(M_m, M_c, M_m_noise, M_c_noise, num_modes, mat_res);
                    y_fin = generate_impact_sound(IR_fin, fs, mass, stiff, vel, 5, 0);
                end
                
                % Normalize and save
                y_normalized = 0.8 * y_fin / max(abs(y_fin));
                audiowrite(savedir_full + num2str(ii) + ".wav", y_normalized, fs);
                
                % Store for randomized version
                trial_idx = trial_idx + 1;
                trial_data{trial_idx}.audio = y_normalized;
                trial_data{trial_idx}.material = char(material);
                trial_data{trial_idx}.size_bucket = size_bucket;
                trial_data{trial_idx}.condition = 2;  % Full model
                trial_data{trial_idx}.iteration = ii;
                trial_data{trial_idx}.source_file = 'full_model';
            end
        end
    end
    
    %% Create randomized experimental stimuli folder for this set
    exp_set_dir = exp_stimuli_dir + "/set_" + string(set_idx);
    if ~exist(exp_set_dir, 'dir')
        mkdir(exp_set_dir);
    end
    
    % Randomize trial order
    num_trials = length(trial_data);
    random_order = randperm(num_trials);
    
    % Prepare metadata table
    trial_num = zeros(num_trials, 1);
    material_col = cell(num_trials, 1);
    size_col = zeros(num_trials, 1);
    condition_col = zeros(num_trials, 1);
    condition_label = cell(num_trials, 1);
    iteration_col = zeros(num_trials, 1);
    source_col = cell(num_trials, 1);
    filename_col = cell(num_trials, 1);
    
    % Save randomized trials and build metadata
    for new_idx = 1:num_trials
        orig_idx = random_order(new_idx);
        trial = trial_data{orig_idx};
        
        % Filename for randomized trial
        trial_filename = sprintf('%03d.wav', new_idx);
        audiowrite(fullfile(char(exp_set_dir), trial_filename), trial.audio, fs);
        
        % Metadata
        trial_num(new_idx) = new_idx;
        material_col{new_idx} = trial.material;
        size_col(new_idx) = trial.size_bucket;
        condition_col(new_idx) = trial.condition;
        if trial.condition == 0
            condition_label{new_idx} = 'recording';
        elseif trial.condition == 1
            condition_label{new_idx} = 'mode_sampling';
        else
            condition_label{new_idx} = 'full_model';
        end
        iteration_col(new_idx) = trial.iteration;
        source_col{new_idx} = trial.source_file;
        filename_col{new_idx} = trial_filename;
    end
    
    % Create and save metadata table
    metadata = table(trial_num, filename_col, material_col, size_col, condition_col, condition_label, iteration_col, source_col, ...
        'VariableNames', {'trial_number', 'filename', 'material', 'size_bucket', 'condition', 'condition_label', 'iteration', 'source_file'});
    writetable(metadata, fullfile(char(exp_set_dir), 'metadata.csv'));
    
    fprintf('  Created randomized set with %d trials\n', num_trials);
end

fprintf('\nDone! Generated %d sets with %d stimuli each.\n', num_sets, length(materials) * length(size_buckets) * num_iterations * 3);
fprintf('Organized stimuli saved to: %s\n', output_base);
fprintf('Randomized stimuli saved to: %s\n', exp_stimuli_dir);

%% Helper function to recursively get all wav files from a directory
function wav_files = get_all_wav_files(base_dir)
    wav_files = {};
    
    % Get all items in the directory
    items = dir(base_dir);
    
    for i = 1:length(items)
        item = items(i);
        
        % Skip . and ..
        if strcmp(item.name, '.') || strcmp(item.name, '..')
            continue;
        end
        
        full_path = fullfile(base_dir, item.name);
        
        if item.isdir
            % Recursively search subdirectories
            sub_files = get_all_wav_files(full_path);
            wav_files = [wav_files; sub_files];
        else
            % Check if it's a wav file
            [~, ~, ext] = fileparts(item.name);
            if strcmpi(ext, '.wav')
                wav_files{end+1, 1} = full_path;
            end
        end
    end
end
