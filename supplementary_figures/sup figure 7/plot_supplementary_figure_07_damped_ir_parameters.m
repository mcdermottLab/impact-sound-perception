close all; clearvars

% Load csv named Amp_1.csv, RT60_1.csv, freq_1.csv, alpha_1.csv,  gamma_1.csv, beta_1.csv from dampIRs_test1 folder

loaddir = 'dampIRs_recs_new7';
objects =["ceramic-damped","ceramic-undamped","glass-damped","glass-undamped","wood-damped","wood-undamped"];
% Load csv files
Amp = csvread(fullfile(loaddir, 'Amp_1.csv'));
RT60 = csvread(fullfile(loaddir, 'RT60_1.csv'));
freq = csvread(fullfile(loaddir, 'freq_1.csv'));
alpha = csvread(fullfile(loaddir, 'alpha_1.csv'));
gamma = csvread(fullfile(loaddir, 'gamma_1.csv'));

directories = readmatrix(fullfile(loaddir, 'directories.csv'));
[dirnums,dir_order] = sort(directories(:,3));

dirnames = csv2cell(loaddir+"/dirnames.csv",'fromfile');

for ii = 1:length(dirnames)
    dirnames1(ii) =dirnames{ii}(end);
    object_mats(ii) = objects(str2double(dirnames1(ii)));

end

all_mats = unique(object_mats);
indd =0;

% Filter frequencies above 20 kHz
freq_cutoff = 20000; % 20 kHz cutoff

for indd = 1: length(all_mats)
    mat = all_mats{indd};
    % Find the indices of the current material
    ind = find(contains(object_mats(dirnums+1),mat));
    RT60_tmp = RT60(dir_order(ind),:);
    freq_tmp = freq(dir_order(ind),:);
    Amp_tmp = Amp(dir_order(ind),:);
    gamma_tmp = gamma(dir_order(ind),:);
    
    % Filter out frequencies above 20 kHz
    valid_freq = freq_tmp <= freq_cutoff;
    RT60_tmp = RT60_tmp(valid_freq);
    freq_tmp = freq_tmp(valid_freq);
    Amp_tmp = Amp_tmp(valid_freq);
    gamma_tmp = gamma_tmp(valid_freq);
    
    RT60_1{indd} = RT60_tmp(:);
    freq_1{indd} = freq_tmp(:);
    Amp_1{indd} = Amp_tmp(:);
    gamma_1{indd} = gamma_tmp(:);

    RT60_mean(indd,:) = mean(RT60_tmp,1);
    freq_mean(indd,:) = mean(freq_tmp,1);
    Amp_mean(indd,:) = mean(Amp_tmp,1);
    gamma_mean(indd,:) = mean(gamma_tmp,1);
    RT60_std(indd,:) = std(RT60_tmp,0,1);
    freq_std(indd,:) = std(freq_tmp,0,1);
    Amp_std(indd,:) = std(Amp_tmp,0,1);
    gamma_std(indd,:) = std(gamma_tmp,0,1);
        
end
    % Find the indices of the current material

figure(1)

for i = 1:length(all_mats)
    errorbar(freq_mean(i,:), RT60_mean(i,:), RT60_std(i,:)/6, 'o', 'LineWidth', 1.5)
    hold on
    legend_str{i} = all_mats{i};
    legend(legend_str)
    set(gca, 'FontSize', 14)
    xlabel('Frequency (Hz)', 'FontSize', 16)
    ylabel('RT60 (s)', 'FontSize', 16)
    title('RT60 vs Frequency for different materials', 'FontSize', 16)
end

figure(2)
iii=1;
for ii = 3:4
    plot(freq_1{ii}, RT60_1{ii}, 'o', 'LineWidth', 1)
    hold on
    legend_str1{iii} = all_mats{ii};
    iii=iii+1;
end
    legend(legend_str1)
    set(gca, 'FontSize',18)
    ylim([0 1])
    xlim([20 20000])
    xlabel('Frequency (Hz)', 'FontSize', 18)
    ylabel('RT60 (s)', 'FontSize', 18)
    title('RT60 vs Frequency for different materials', 'FontSize', 16)

% New figure for Amplitude vs Frequency
figure(3)
iii=1;
for ii = 3:4
    plot(freq_1{ii}, Amp_1{ii}, 'o', 'LineWidth', 1)
    hold on
    legend_str2{iii} = all_mats{ii};
    iii=iii+1;
end
    ylim([-40 10])
    legend(legend_str2)
    set(gca, 'FontSize',18)
    xlabel('Frequency (Hz)', 'FontSize', 18)
    ylabel('Amplitude', 'FontSize', 18)
    title('Amplitude vs Frequency for different materials', 'FontSize', 16)
    xlim([20 20000])
    
% Create fig_damp folder if it doesn't exist
if ~exist('fig_damp', 'dir')
    mkdir('fig_damp');
end

saveas(2, '../../Figures/all_figs/SupFigure7_1.svg')

saveas(3, '../../Figures/all_figs/SupFigure7_2.svg')