clearvars; close all;

cd("../")
soundFiles = dir('../data/15_modes_v5/observed/**/*.wav');
numFiles = length(soundFiles);

% Parameters
windowSize = 0.02; % 20ms
fs = 44100; % Assuming standard sampling rate, adjust if different
samplesPerWindow = round(windowSize * fs);

% Initialize arrays to store spectra
firstWindowSpectra = [];
secondWindowSpectra = [];
thirdWindowSpectra = [];
fourthWindowSpectra = [];

% Process each sound file
for i = 1:numFiles
    [y, fs] = audioread(fullfile(soundFiles(i).folder, soundFiles(i).name));

    y = y(1:end);
    
    % Extract windows
    firstWindow = y(1:samplesPerWindow);
    secondWindow = y(samplesPerWindow+1:2*samplesPerWindow);
    thirdWindow = y(2*samplesPerWindow+1:3*samplesPerWindow);
    
    % Only extract fourth window if enough samples are available
    if length(y) >= 4*samplesPerWindow
        fourthWindow = y(3*samplesPerWindow+1:4*samplesPerWindow);
    else
        fourthWindow = [];
    end
    
    % Calculate spectra (using FFT)
    nfft = 2^nextpow2(samplesPerWindow);
    first_spectrum = abs(fft(firstWindow, nfft));
    second_spectrum = abs(fft(secondWindow, nfft));
    third_spectrum = abs(fft(thirdWindow, nfft));
    
    % Only calculate fourth spectrum if we have the data
    if ~isempty(fourthWindow)
        fourth_spectrum = abs(fft(fourthWindow, nfft));
        fourth_spectrum = fourth_spectrum(1:nfft/2+1);
        fourthWindowSpectra(:,i) = fourth_spectrum;
    else
        fourth_spectrum = nan(nfft/2+1, 1);
        fourthWindowSpectra(:,i) = fourth_spectrum;
    end
    
    % Keep only positive frequencies and normalize
    first_spectrum = first_spectrum(1:nfft/2+1);
    second_spectrum = second_spectrum(1:nfft/2+1);
    third_spectrum = third_spectrum(1:nfft/2+1);
    fourth_spectrum = fourth_spectrum(1:nfft/2+1);
    
    % Store spectra
    firstWindowSpectra(:,i) = first_spectrum;
    secondWindowSpectra(:,i) = second_spectrum;
    thirdWindowSpectra(:,i) = third_spectrum;
    fourthWindowSpectra(:,i) = fourth_spectrum;
end

% Calculate frequency axis
freq = linspace(0, fs/2, length(first_spectrum));

% Calculate average spectra
avgFirstSpectrum = mean(firstWindowSpectra, 2);
avgSecondSpectrum = mean(secondWindowSpectra, 2);

% Calculate spectral flatness for both windows
flatness_first = geomean(avgFirstSpectrum) / mean(avgFirstSpectrum);
flatness_second = geomean(avgSecondSpectrum) / mean(avgSecondSpectrum);

% Calculate spectral flatness for each file individually
flatness_first_all = zeros(numFiles, 1);
flatness_second_all = zeros(numFiles, 1);
flatness_third_all = zeros(numFiles, 1);
flatness_fourth_all = zeros(numFiles, 1);
for i = 1:numFiles
    flatness_first_all(i) = geomean(firstWindowSpectra(:,i)) / mean(firstWindowSpectra(:,i));
    flatness_second_all(i) = geomean(secondWindowSpectra(:,i)) / mean(secondWindowSpectra(:,i));
    flatness_third_all(i) = geomean(thirdWindowSpectra(:,i)) / mean(thirdWindowSpectra(:,i));
    flatness_fourth_all(i) = geomean(fourthWindowSpectra(:,i)) / mean(fourthWindowSpectra(:,i));
end

% Calculate pairwise differences between flatness values
pairwise_differences = flatness_first_all - flatness_second_all;

% Remove NaN values before calculating statistics
flatness_first_all = flatness_first_all(~isnan(flatness_first_all));
flatness_second_all = flatness_second_all(~isnan(flatness_second_all));
flatness_third_all = flatness_third_all(~isnan(flatness_third_all));
flatness_fourth_all = flatness_fourth_all(~isnan(flatness_fourth_all));
pairwise_differences = pairwise_differences(~isnan(pairwise_differences));

% Calculate means and standard deviations
flatness_first = mean(flatness_first_all);
flatness_second = mean(flatness_second_all);
flatness_third = mean(flatness_third_all);
flatness_fourth = mean(flatness_fourth_all);
std_first = std(flatness_first_all);
std_second = std(flatness_second_all);
std_third = std(flatness_third_all);
std_fourth = std(flatness_fourth_all);

% Calculate SEM
sem_first = std(flatness_first_all) / sqrt(length(flatness_first_all));
sem_second = std(flatness_second_all) / sqrt(length(flatness_second_all));
sem_third = std(flatness_third_all) / sqrt(length(flatness_third_all));
sem_fourth = std(flatness_fourth_all) / sqrt(length(flatness_fourth_all));

% Create a new figure for four-window flatness comparison with size in inches (4x4)
figure('Units', 'inches', 'Position', [1 1 4 4]);

% Prepare data for error bars
means = [flatness_first, flatness_second, flatness_third, flatness_fourth];
sems = [sem_first, sem_second, sem_third, sem_fourth];
x = [20, 40, 60, 80];  % Time points (end of each window)

% Create error bar plot with black line and no cap
errorbar(x, means, sems, 'o-', 'LineWidth', 2, 'MarkerSize', 5, 'CapSize', 0, 'Color', 'k', 'MarkerFaceColor', 'k');
hold on;

% Add individual data points with jitter
jitter_width = 2;
scatter(20 + jitter_width*(rand(size(flatness_first_all))-0.5), flatness_first_all, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.01);
scatter(40 + jitter_width*(rand(size(flatness_second_all))-0.5), flatness_second_all, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.01);
scatter(60 + jitter_width*(rand(size(flatness_third_all))-0.5), flatness_third_all, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.01);
scatter(80 + jitter_width*(rand(size(flatness_fourth_all))-0.5), flatness_fourth_all, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.01);

% Redraw error bars on top
errorbar(x, means, sems, 'o-', 'LineWidth', 2, 'MarkerSize', 5, 'CapSize', 0, 'Color', 'k', 'MarkerFaceColor', 'k');

% Customize the plot
xlabel('Time (ms)', 'FontSize', 12);
ylabel('Spectral Flatness', 'FontSize', 12);
title('Spectral Flatness vs Time', 'FontSize', 12);
xlim([0 100]);
ax = gca;
ax.FontSize = 12;

% Save updated figure as SVG
cd("Figure 2")
saveas(gcf,'../all_figs/Figure_2f.svg');

