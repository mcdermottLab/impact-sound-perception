close all
% Generate a single sinusoidal bump
x = linspace(0, pi, 1000);
y = sin(x);
y(x < 0 | x > pi) = 0;  % Zero out everything outside the bump

% Create the soft-clipped version using tanh
clip_factor =0.1;
y_clipped = 1/clip_factor*tanh(clip_factor * y);

% Create the figure
figure('Units', 'inches', 'Position', [2, 2, 4, 4]);

hold on;

% Plot the soft-clipped version (solid red line)
plot(x, y_clipped, 'r', 'LineWidth', 2);

% Customize the plot
xlabel('x');
ylabel('y');
axis([0 1.5*pi -1.1 1.1]);

% Save the figure as a vector graphic
saveas(gcf, 'sinusoidal_bump_comparison.svg');

% Create the time-domain bump signal
t = linspace(0, 1, 44100);
bump_duration = 0.001; % 5ms duration
bump_samples = round(bump_duration * 44100);
t_bump = linspace(0, pi, bump_samples);
y_clipped = zeros(1, 44100);
y_clipped(10000:9999+bump_samples) = sin(t_bump);

% Compute FFT
Fs = 44100;  % Sampling frequency
L = length(y_clipped);
Y = fft(y_clipped);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

% Create frequency axis
f = Fs*(0:(L/2))/L;

% Create new figure for frequency domain
figure('Units', 'inches', 'Position', [2, 2, 4, 4]);

% Plot amplitude spectrum on log scale
plot(f, P1, 'r', 'LineWidth', 2);

% Customize the frequency plot
xlabel('Frequency (Hz)');
ylabel('Amplitude (log scale)');
xlim([0 10000]);

% Save the frequency domain plot
saveas(gcf, 'sinusoidal_bump_fft.svg');

% Create shifted sinc function
figure('Units', 'inches', 'Position', [2, 2, 4, 4]);
t_sinc = linspace(-10, 10, 1000);
shift = 0;
sinc_func = abs(sinc(t_sinc - shift));  % Take absolute value

% Plot the shifted sinc on log scale
semilogy(t_sinc, sinc_func, 'r', 'LineWidth', 2);

% Customize the plot
xlabel('x');
ylabel('y');
axis([-10 10 1e-2 1.1]);

% Save the sinc plot
saveas(gcf, 'shifted_sinc.svg');
