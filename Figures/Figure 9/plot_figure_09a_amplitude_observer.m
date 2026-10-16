clearvars; close all;

load("model_10dBnoise_0.04_amp_rms_400_v4.mat")

damps_cond = [0,0.25,0.5,0.75,0.9,0.99]; conds_name = {'Recorded','Full generative model','Inverted covariance','Scaled Variance','Uniform model','Inverted Distribution','Linear decay'};

% number of tries x damping conds x model conds
mod_ans_cond = 1 - mean(mod_ans_all(:,:,:,:), 4);

mean_perf = squeeze(mean(mod_ans_cond, 1));
std_perf = squeeze(std(mod_ans_cond, 0, 1)) / sqrt(size(mod_ans_cond, 1));

conds_use = [1 2 3 4 5 6 7];

% Create figure outside the loop
figure
% set the size of the figure so it is readable in the paper
set(gcf, 'Units', 'inches', 'Position', [1, 1, 5, 2.5])

% Create small x-axis offsets for each condition to separate the lines
x_positions = 1:5;
offset_step = 0.0;
offsets = (-(length(conds_use)-1)/2 : (length(conds_use)-1)/2) * offset_step;

for i = 1:length(conds_use)
    condss = conds_use(i);
    % Add offset to x positions for each condition
    x_with_offset = x_positions + offsets(i);
    errorbar(x_with_offset, mean_perf(:,condss), std_perf(:,condss), 'LineWidth', 3, 'Capsize', 0)
    hold on
end

set(gca, 'FontSize', 12)
set(gca, "XTick", 1:5)
set(gca, 'XTickLabel', {'1.8','3.16','10','31.6','100'})
xlabel('Mass ratio')
ylabel('Proportion correct')
ylim([0.45 1.01])
xlim([0 6])
hold on
% draw a dashed line to show chance performance
plot([0 1.2], [0.5 0.5], '--', 'LineWidth', 2, 'Color', 'k')
hold off
saveas(gcf, "../all_figs/Figure_9a.svg")

fprintf('Plot saved to ../all_figs/Figure_9a.svg\n');

