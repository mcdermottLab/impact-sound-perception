clearvars; close all
load('power_explained_means_db.mat');

% Display the means
fprintf('Mean power explained (all modes) fraction: %f\n', mean_power_explained_all);
fprintf('Mean power explained (top 10 modes) fraction: %f\n', mean_power_explained_top10);
fprintf('Mean power explained (top 5 modes) fraction: %f\n', mean_power_explained_top5);
fprintf('Mean power explained (top 2 modes) fraction: %f\n', mean_power_explained_top2);
fprintf('Mean power explained (20 modes) fraction: %f\n', mean_power_explained_20modes);

% Create figure with five subplots
figure('Position', [100, 100, 2000, 400]);

% Common plotting parameters
num_bins = 50;
plot_params.xlabel = 'Fraction of Power Explained';
plot_params.ylabel = 'Probability';
plot_params.xlim = [0, 1];
plot_params.fontsize = 16;

% Create the subplots in a loop
data_to_plot = {
    power_explained_20modes, mean_power_explained_20modes, '(20 Modes)';
    power_explained_all, mean_power_explained_all, '(15 Modes)';
    power_explained_top10, mean_power_explained_top10, '(Top 10 Modes)';
    power_explained_top5, mean_power_explained_top5, '(Top 5 Modes)';
    power_explained_top2, mean_power_explained_top2, '(Top 2 Modes)'
};

for i = 1:5
    subplot(1, 5, i);
    histogram(data_to_plot{i, 1}, num_bins, 'Normalization', 'probability');
    hold on;
    line([data_to_plot{i, 2}, data_to_plot{i, 2}], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--');
    text(data_to_plot{i, 2}, 0.95*max(ylim), sprintf('Mean: %.2f', data_to_plot{i, 2}), ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontSize', 14, 'Color', 'r');
    hold off;
    title(data_to_plot{i, 3}, 'FontSize', 18);
    xlabel(plot_params.xlabel, 'FontSize', plot_params.fontsize);
    ylabel(plot_params.ylabel, 'FontSize', plot_params.fontsize);
    xlim(plot_params.xlim);
    ax = gca;
    ax.FontSize = plot_params.fontsize;
end

% Adjust layout and save figure
sgtitle('Power Explained Distributions (0 ≤ power explained ≤ 1)', 'FontSize', 20);
print('power_explained_histograms', '-dpng', '-r300');

% Create new figure for error bar plot with size in inches (5x5)
figure('Units', 'inches', 'Position', [2, 2, 4, 4]);

% Prepare data for error bars
means = [mean_power_explained_top2, mean_power_explained_top5, mean_power_explained_top10, mean_power_explained_all, mean_power_explained_20modes];
sems = [nanstd(power_explained_top2)/sqrt(length(power_explained_top2)), ...
        nanstd(power_explained_top5)/sqrt(length(power_explained_top5)), ...
        nanstd(power_explained_top10)/sqrt(length(power_explained_top10)), ...
        nanstd(power_explained_all)/sqrt(length(power_explained_all)), ...
        nanstd(power_explained_20modes)/sqrt(length(power_explained_20modes))];
x = [2, 5, 10, 15, 20];  % Number of modes

% Create error bar plot with black line and no cap
errorbar(x, means, sems, 'o-', 'LineWidth', 2, 'MarkerSize', 5, 'CapSize', 0, 'Color', 'k', 'MarkerFaceColor', 'k');
hold on;

% Add individual data points with jitter
jitter_width = 0.5;
% Plot individual points for each mode count
scatter(2 + jitter_width*(rand(size(power_explained_top2))-0.5), power_explained_top2, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
scatter(5 + jitter_width*(rand(size(power_explained_top5))-0.5), power_explained_top5, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
scatter(10 + jitter_width*(rand(size(power_explained_top10))-0.5), power_explained_top10, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
scatter(15 + jitter_width*(rand(size(power_explained_all))-0.5), power_explained_all, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
scatter(20 + jitter_width*(rand(size(power_explained_20modes))-0.5), power_explained_20modes, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
errorbar(x, means, sems, 'o-', 'LineWidth', 2, 'MarkerSize', 5, 'CapSize', 0, 'Color', 'k', 'MarkerFaceColor', 'k');
% Customize the plot
xlabel('Number of Modes', 'FontSize', 12);
ylabel('Fraction of Power Explained', 'FontSize', 12);
title('Power Explained vs Number of Modes', 'FontSize', 12);
xlim([0 22])
ax = gca;
ax.FontSize = 12;

% Save the new figure
saveas(gcf,'../all_figs/Figure_2d.svg');

