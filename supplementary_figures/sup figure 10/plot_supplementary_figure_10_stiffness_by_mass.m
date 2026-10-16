clearvars; close all;

% Add path to synthesis_model for generate_impact_sound function
addpath('../../synthesis_model');

savedir = "./test_set_v2/";

% Wood materials
wood_materials = ["wood_mass1m", "wood_mass2m", "wood_mass3m", "wood_mass4m"];
wood_masses = [1.2 5.9 8.7 12]*10^-2;

% Metal materials
metal_materials = ["metal_large_1", "metal_large_2", "metal_large_3", "metal_large_4"];
metal_masses = [1.2 5.9 8.7 12]*10^-2;  % Assuming masses, please adjust if different

% Combine materials and masses
materials = [wood_materials, metal_materials];
masses = [wood_masses, metal_masses];

IR_calib = audioread("calib.wav");
IR_calib = IR_calib(:,1);
IR_calib = IR_calib/max(abs(IR_calib));
% Extract and smoothen the spectrum of the calibration impulse response
IR_calib_fft = abs(fft(IR_calib));
IR_calib_rfft = IR_calib_fft(1:ceil(length(IR_calib_fft)/2));
IR_calib_rfft = smoothdata(IR_calib_rfft,"movmean",2000);

% Get all wav files in the directory ./wood_mass2m/
load_dir = savedir + materials{1} + "/";
files = dir(load_dir + "*.wav");

min_mass_all = cell(length(materials), 1);

for material_idx = 1:length(materials)
    material = materials(material_idx);
    
    % Set the appropriate load directory
    if contains(material, 'wood')
        load_dir = savedir + material + "/";
    else
        load_dir = "./metal_mass/" + material + "/";
    end
    files = dir(load_dir + "*.wav");

    min_mass_all{material_idx} = [];

    for file_idx = 1:length(files)
        [y, Fs] = audioread(load_dir + files(file_idx).name);
        if size(y,2) > 0
            y = y(:,1);
        end
        y_fin = y;

        % Find the IR file
        if contains(material, 'wood')
            IR_dir = load_dir + "IRs/";
            IR_files = dir(IR_dir + "observed_*.wav");
            if isempty(IR_files)
                IR_files = dir(IR_dir + "h*.wav");
            end
            if isempty(IR_files)
                warning('No IR file found for: %s', files(file_idx).name);
                continue;
            end
            IR_fin = audioread(IR_dir + IR_files(1).name);
        else
            % For metal, use the single IR file
            IR_fin = audioread("./metal_mass/h.wav");
        end

        mass = masses(material_idx);

        if contains(material, 'wood')
            stiff = logspace(7, 8, 200);
        else
            stiff = logspace(8, 9, 200);
        end
        vel = 5;

            y_test = y_fin;

            % Calculate the FFT of the test signal
            y_test_fft = abs(fft(y_test));
            y_test_rfft = y_test_fft(1:ceil(length(y_test_fft)/2));
            y_test_rfft = y_test_rfft ./ IR_calib_rfft(1:length(y_test_rfft));
            jkl=1;
            for ii = 1:length(stiff)
                stiff_cand = stiff(ii);
                y_cand = generate_impact_sound(IR_fin, 44100, mass, stiff_cand, vel, 5, 0);
                
                if length(y_test)<length(y_cand)
                y_cand = y_cand(1:length(y_test));
                else
                  y_test = y_test(1:length(y_cand));  
                end
                y_cand = y_cand/rms(abs(y_cand));
                y_test = y_test/rms(abs(y_test));
            
                % calculate the loss function between y_cand and y_test in the fourier domain
                y_cand_fft = abs(fft(y_cand));
                y_test_fft = abs(fft(y_test));
                
                % smoothen both the ffts with a gaussian window of length 1000
                
                y_cand_rfft =y_cand_fft(1:ceil(length(y_cand_fft)/2));
                y_test_rfft =y_test_fft(1:ceil(length(y_test_fft)/2));
                IR_calibed = interp1(1:length(IR_calib_rfft),IR_calib_rfft,1:length(y_cand_rfft));
            
                % Take first half of the rffts and take their log
                if length(y_cand_rfft)>5000
                    y_cand_rfft = log(y_cand_rfft(1:5000));
                    y_test_rfft = log(y_test_rfft(1:5000));
                else
                    y_cand_rfft = log(y_cand_rfft);
                    y_test_rfft = log(y_test_rfft);
                end

                loss(ii) = sum(abs(y_cand_rfft-y_test_rfft).^2);
                if ii == 1
                    min_loss = loss(ii);
                    min_mass = stiff(ii);
                end
                if min_loss > loss(ii)
                    min_loss = loss(ii);
                    min_mass = stiff(ii);
                end
            end
            min_mass_all{material_idx}(file_idx, jkl) = min_mass;
       
    end
end

% Calculate mean and std stiffness from min_mass_all
mean_stiffness = zeros(1, length(materials));
std_stiffness = zeros(1, length(materials));
for material_idx = 1:length(materials)
    data = min_mass_all{material_idx}(:);
    data = data(data > 0);  % Remove zeros (from unprocessed files)
    if ~isempty(data)
        mean_stiffness(material_idx) = mean(data);
        std_stiffness(material_idx) = std(data);
    end
end

% Add a new figure for error bar plot
figure(3)
hold on

% Separate wood and metal data
wood_mean_stiffness = mean_stiffness(1:length(wood_masses));
wood_std_stiffness = std_stiffness(1:length(wood_masses));
metal_mean_stiffness = mean_stiffness(length(wood_masses)+1:end);
metal_std_stiffness = std_stiffness(length(wood_masses)+1:end);

% Plot wood data
errorbar(wood_masses*100, wood_mean_stiffness, wood_std_stiffness, 'o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', 'b', 'DisplayName', 'Wood (inferred)')

% Plot metal data
errorbar(metal_masses*100, metal_mean_stiffness, metal_std_stiffness, 's', 'LineWidth', 2, 'MarkerSize', 10, 'Color', 'r', 'DisplayName', 'Metal (inferred)')

% Add blue dashed line at 6.4*10^7 N/m
yline(6.4e7, '--b', 'LineWidth', 2, 'DisplayName', 'Wood (theory)')

% Add red dashed line at 2.03*10^8 N/m
yline(2.03e8, '--r', 'LineWidth', 2, 'DisplayName', 'Metal (theory)')

xlabel("Mass (g)")
ylabel("Stiffness (N/m)")
title("Average Stiffness with Error Bars for Wood and Metal Masses")

% Set X-axis ticks to wood masses
xticks(wood_masses*100)
xticklabels(arrayfun(@(x) sprintf('%.1f', x), wood_masses*100, 'UniformOutput', false))

set(gca, 'YScale', 'log')
set(gca, 'YLim', [1e7 1e9])  % Set y-axis limits
set(gca, 'FontSize', 24)
set(gcf, 'Position', [100 100 1500 800])
grid on
legend('Location', 'best')
hold off

saveas(gcf, "../../Figures/all_figs/sup_figure_10.svg")
