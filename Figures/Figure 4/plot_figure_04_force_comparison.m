clearvars; close all; 

load_dir = "./wood_sounds/";

% Load the wood_IR.wav file
[y, Fs] = audioread(load_dir +"wood_IR.wav");
y = y(find(y, 1):end); % Remove zeros from the beginning
y1 = y;

% Calculate the position y first becomes 0.001
start_index = find(abs(y) > 0.001, 1);
y = y / max(abs(y));
y = y(start_index:end);

% Load and process the wood_smallball.wav file
[y_smallball, Fs_smallball] = audioread(load_dir +"wood_smallball.wav");
y_smallball = y_smallball / max(abs(y_smallball)); % Normalize
start_index = find(abs(y_smallball) > 0.001, 1);
y_smallball = y_smallball(start_index:end);

% Load and process the wood_largeball.wav file
[y_largeball, Fs_largeball] = audioread(load_dir +"wood_largeball3.wav");

start_index = find(abs(y_largeball) > 0.001, 1);
y_largeball = y_largeball(start_index:end);
y_largeball = y_largeball / max(abs(y_largeball)); % Normalize

plot_spectrogram(y, Fs);

mass1 = 0.012;
 mass2 = 0.058;
stiff = 0.5*10^7;

imp1 = generate_impact_sound(y,Fs,mass1,stiff,4,4,0);
imp2 = generate_impact_sound(y,Fs,mass2,stiff,4,4,0);

% Use the first 0.2 seconds of y_smallball and imp1
duration = 0.15;
samples = round(duration * Fs);

y_smallball_short = y_smallball(1:min(samples, length(y_smallball)));
y_smallball_short = (mass1/mass2)*y_smallball_short/max(abs(y_smallball_short));
y_largeball_short = y_largeball(1:min(samples, length(y_largeball)));
y_short = y(1:min(samples, length(y)));
imp1_short = (mass1/mass2)*imp1(1:min(samples, length(imp1)))/max(abs(imp1));
imp2_short = imp2(1:min(samples, length(imp2)));
imp2_short = imp2_short -mean(imp2_short);
imp2_short = imp2_short/max(abs(imp2_short));

y_largeball_short = y_largeball_short / max(abs(y_largeball_short));
y_short = y_short*(mass1/mass2);

% Create a new figure for the large ball and its simulation
figure('Units', 'inches', 'Position', [1 1 5 4]);  % 5in x 5in figure

% Plot large ball recording and simulation

plot_linear_spectrum(y_short, Fs, 'Small Ball Recording', '-', [0.6 0.2 0.4]); % Muted burgundy
hold on;
plot_linear_spectrum(y_largeball_short, Fs, 'Large Ball Recording', '-', [0.2 0.4 0.4]); % Muted teal

set(gca, 'FontSize', 12);

xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Spectrum Comparison: Large Ball');
legend('Small Ball', 'Large Ball');
xlim([0 14000]);  % Limit x-axis to audible frequency range
ylim([-120 -40]);

% Save the spectrum comparison figure as a vector graphic
saveas(gcf, '../all_figs/Figure_4b.svg');

% Create figs directory if it doesn't exist
if ~exist('figs', 'dir')
    mkdir('figs');
end

% Plot and save spectrum for small ball recording
figure('Position', [10 50 800 600]);
plot_spectrogram(y_short, Fs);
title('Spectrum: Small Ball Recording', 'FontSize', 12);
saveas(gcf, '../all_figs/Figure_4c_1.svg');

% Plot and save spectrum for small ball simulation
figure('Position', [10 50 800 600]);
plot_spectrogram(imp1_short, Fs);
title('Spectrum: Small Ball Simulation', 'FontSize', 12);
saveas(gcf, '../all_figs/Figure_4e_1.svg');

% Plot and save spectrum for large ball recording
figure('Position', [10 50 800 600]);
plot_spectrogram(y_largeball_short, Fs);
title('Spectrum: Large Ball Recording', 'FontSize', 12);
saveas(gcf, '../all_figs/Figure_4c_2.svg');

% Plot and save spectrum for large ball simulation
figure('Position', [10 50 800 600]);
plot_spectrogram(imp2_short, Fs);
title('Spectrum: Large Ball Simulation', 'FontSize', 12);
saveas(gcf, '../all_figs/Figure_4e_2.svg');

% Function to plot linear spectrum
function plot_linear_spectrum(signal, Fs, label, linestyle, color)
    N = length(signal);
    f = (0:N-1)*(Fs/N);
    Y = fft(signal);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    % Convert to dB
    P1_db = 20*log10(P1);
    
    % Apply moving average filter for smoothing
    window_size = 20;
    P1_smoothed = movmean(P1_db, window_size);
    plot(f(1:N/2+1), P1_smoothed, linestyle, 'DisplayName', label, 'LineWidth', 2, 'Color', color);
end