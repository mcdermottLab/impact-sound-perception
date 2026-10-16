clearvars; close all;

load_dir = "./all_recs/";
% Get a list of all WAV files in the current directory
fileList = dir(load_dir +"*.wav");

material_list = [ "glass", "ceramic", "metal","stone","wood","plastic", "cardboard"];

% Initialize a matrix to store the maximum amplitudes
maxAmplitudes = [];
maxAmps = NaN([length(fileList) 40]);

% Loop through each WAV file
for i = 1:length(fileList)
    close all
    y_fin = []; maxAmplitudes =[]; maxAmplitudesPerFile =[];
    % Load the WAV file
    [audio, fs] = audioread(load_dir+fileList(i).name);

    savedir= load_dir + string(fileList(i).name(1:end-4))+"/";
    if ~exist(savedir, 'dir')
        mkdir(savedir);  % create the directory if it doesn't exist
    end

        count = 0;
        y= audio;
        sq_y = smoothdata(y.^2,"gaussian",400);
        %Find indices where sq_y increases suddenly
        diff_y = diff(sq_y)*44100;
        [pks,locs]= findpeaks(diff_y,'MinPeakHeight',0.001,MinPeakDistance=30000);

        for kk = 1:length(locs)
            y_fin1 = y((locs(kk)-3000):(locs(kk)+15000));
            
            y_fin{kk} = y_fin1;
            count = count+1;
            maxAmplitudesPerFile(count)=max(abs(y_fin1));
            audiowrite(savedir+count+".wav",y_fin{kk},fs);
        end
    
    % Append the vector to the matrix
    maxAmplitudes{i} = 20*log10(maxAmplitudesPerFile/2e-6);
    ave_Amp(i) = mean(maxAmplitudes{i});
    std_Amp(i) = std(maxAmplitudes{i});
    maxAmps(i,1:length(maxAmplitudesPerFile)) = 20*log10(maxAmplitudesPerFile/2e-6);
    
end

removeLastFour = @(s) s(1:end-4);

newCellArray = cellfun(removeLastFour, {fileList.name}, 'UniformOutput', false);
   
% Plot the errorbars for the maximum amplitudes
figure;

% Extract material names from file names
materials = cellfun(@(x) strsplit(x, '_'), newCellArray, 'UniformOutput', false);
materials = cellfun(@(x) x{1}, materials, 'UniformOutput', false);

% Use the predefined material_list for ordering
material_list = ["glass", "ceramic", "metal", "wood", "stone","plastic", "cardboard"];

% Create a colormap
colorMap = jet(length(material_list));

% Plot the errorbars for the maximum amplitudes
figure;
% Set figure size to be larger
figure('Position', [100, 100, 1200, 800]);

hold on;
count = 0;
count1 = 1;
file_order = [];
low_amp_indices = [];
all_indices = 1:length(materials);
e1 = gobjects(1, length(material_list)); % Preallocate e1 array

for i = 1:length(material_list)
    material = material_list(i);
    materialIndices = strcmp(materials, material);
    
    if any(materialIndices)
        % Separate low amplitude files
        low_amp = ave_Amp(materialIndices) < 85;
        indices = all_indices(materialIndices);

        if any(find(low_amp))
            low_amp_indices = [low_amp_indices indices(low_amp)];
        end
        % Plot only files with amplitude >= 85 dB
        high_amp_indices = indices(~low_amp);
        if ~isempty(high_amp_indices)
            e1(i) = errorbar((1+count):(count+length(high_amp_indices)), ave_Amp(high_amp_indices), std_Amp(high_amp_indices), 'o', 'LineWidth', 2, 'Color', colorMap(count1,:), 'DisplayName', material);
        end
        file_order = [file_order high_amp_indices];
        count = count + length(high_amp_indices);
    end
    count1 = count1 + 1;
end

% Remove empty entries from e1
e1 = e1([1:4 6:7]);

% Add legend for e1 errorbars (only for high amplitude sounds)
legend(e1, 'Location', 'northeast', 'FontSize', 18);

% Plot low amplitude files at the end
for i = 1:length(material_list)
    material = material_list(i);
    low_amp_material = low_amp_indices(strcmp(materials(low_amp_indices), material));
    
    if ~isempty(low_amp_material)
        h = errorbar((1+count):(count+length(low_amp_material)), ave_Amp(low_amp_material), std_Amp(low_amp_material), 'o', 'LineWidth', 2, 'Color', colorMap(i,:));
        h.Bar.LineStyle = 'dotted';  % Set error bar ticks to dashed
        file_order = [file_order low_amp_material];
        count = count + length(low_amp_material);
    end
end

xlabel('Sound label');
ylabel('Maximum amplitude (dB)');
title('Maximum amplitudes for recorded impact sounds');
set(gca, 'XTick', 1:length(file_order), 'XTickLabel', newCellArray(file_order), "TickLabelInterpreter", "none");
xlim([0 length(file_order)+2]);
ylim([60 120]);
set(gca, 'FontSize', 18);
grid on;
grid minor;

% Save the plot to a svg file
saveas(gcf, '../../Figures/all_figs/SupFigure9.svg');
