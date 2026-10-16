clearvars; close all;
load("Exp4_data.mat")

materials = ["ceramic","glass","metal","stone","wood"];

damps_cond = [0,0.25,0.5,0.75,0.90];conds_name = {'Recorded','Full generative model','Inverted covariance','Scaled Variance','Uniform distribution','linear decay'};
conds_use = 1:6;

for size_p = 1:length(damps_cond)

resp_0cond = squeeze(resp_cond(:,:,1,:));
% create a variable corr_0cond that is 1 whereever resp_0cond is 0 and 0 elsewhere
corr_0cond = resp_0cond==1;
resp_1cond = squeeze(resp_cond(:,:,2:end,:));
corr_1cond = resp_1cond==2 | resp_1cond==3;

hit_rate = squeeze(sum(corr_1cond,[1 4])/(size(corr_1cond,4)*size(corr_1cond,1)));
cr_rate = sum(corr_0cond,[1 3])/(size(corr_0cond,3)*size(corr_0cond,1));
fp_rate = 1-cr_rate;
% Calculate the d' for each participant
d_prime = norminv(hit_rate)'-norminv(fp_rate);
dp_max = 2 ;
d_prime(d_prime>dp_max) =dp_max;

% Get participant wise d'
hit_rate_part = squeeze(sum(corr_1cond,[4])/(size(corr_1cond,4)));
cr_rate_part = sum(corr_0cond,[3])/(size(corr_0cond,3));
fp_rate_part = 1-cr_rate_part;

% Calculate the d' for each participant
for part_ind = 1:size(corr_0cond,1)
    d_prime_part(part_ind,:,:) = norminv(squeeze(hit_rate_part(part_ind,:,:)))'-norminv(fp_rate_part(part_ind,:));
    
    if any(isnan(d_prime_part(part_ind,:,:)))
        fprintf('NaN found for participant %d\n', part_ind);
        disp('Hit rates:');
        disp(squeeze(hit_rate_part(part_ind,:,:)));
        disp('False positive rate:');
        disp(fp_rate_part(part_ind,:));
    end
end

d_prime_part(d_prime_part>dp_max) =dp_max;
std_dprime = squeeze(nanstd(d_prime_part,0,1)/sqrt(length(d_prime_part)));

end

% calculate the d' over hit_rate_all and fp_rate_all
hit_rate_all = hit_rate;
fp_rate_all = fp_rate;
d_prime_all = norminv(hit_rate_all)'-norminv(fp_rate_all);

% calculate the d' over hit_rate_all_part and fp_rate_all_part
hit_rate_all_part = hit_rate_part;
fp_rate_all_part = fp_rate_part;

hit_rate_all_part(hit_rate_all_part == 1) = 0.99;
hit_rate_all_part(hit_rate_all_part == 0) = 0.01;
fp_rate_all_part(fp_rate_all_part == 1) = 0.99;
fp_rate_all_part(fp_rate_all_part == 0) = 0.01;

% Bootstrap analysis for d_prime_all
n_bootstrap = 10000;
bootstrap_d_prime = zeros(n_bootstrap, size(hit_rate_all_part,2), size(hit_rate_all_part,3));

for b = 1:n_bootstrap
    % Bootstrap resample participants
    bootstrap_indices = randsample(size(hit_rate_all_part,1), max(10,randi(size(hit_rate_all_part,1))), true);
    
    % Calculate mean hit rates and false positive rates for this bootstrap sample
    bootstrap_hit_rates = squeeze(mean(hit_rate_all_part(bootstrap_indices,:,:), 1));
    bootstrap_fp_rates = squeeze(mean(fp_rate_all_part(bootstrap_indices,:), 1));
    
    % Apply the same corrections as before
    bootstrap_hit_rates(bootstrap_hit_rates == 1) = 0.95;
    bootstrap_hit_rates(bootstrap_hit_rates == 0) = 0.05;
    bootstrap_fp_rates(bootstrap_fp_rates == 1) = 0.95;
    bootstrap_fp_rates(bootstrap_fp_rates == 0) = 0.05;
    
    % Calculate d' for this bootstrap sample
    bootstrap_d_prime(b,:,:) = norminv(bootstrap_hit_rates) - norminv(bootstrap_fp_rates)';
end

std_dprime_all = squeeze(std(bootstrap_d_prime, 0, 1))';

for part_ind = 1:size(hit_rate_all_part,1)
    d_prime_all_part(part_ind,:,:) = norminv(squeeze(hit_rate_all_part(part_ind,:,:)))'-norminv(fp_rate_all_part(part_ind,:));
    
    if any(isnan(d_prime_all_part(part_ind,:,:)))
        fprintf('NaN found for participant %d in all materials calculation\n', part_ind);
        disp('Hit rates:');
        disp(squeeze(hit_rate_all_part(part_ind,:,:)));
        disp('False positive rate:');
        disp(fp_rate_all_part(part_ind,:));
    end
end

d_prime_all_part(d_prime_all_part>dp_max) =dp_max;

d_prime_all(d_prime_all>dp_max) =dp_max;

figure(2)
set(gcf, 'Units', 'inches', 'Position', [0, 0, 5, 4]);

% Create perturbed x-positions
x_positions = repmat(damps_cond(2:end)', 1, size(d_prime_all, 2));
perturbation = linspace(-0.01, 0.01, size(d_prime_all, 2));
x_perturbed = x_positions + perturbation;

% Plot errorbars with perturbed x-positions
for i = 1:size(d_prime_all, 2)
    errorbar(x_perturbed(:,i), d_prime_all(:,i), std_dprime_all(:,i), '-o', 'LineWidth', 2, 'MarkerSize', 5,'CapSize',0);
    hold on
end

title("All materials");
xlabel('Damping ratio');
ylabel('d''');
ylim([0,2.4]);
xlim([min(damps_cond(2:end))-0.05, max(damps_cond(2:end))+0.05]);
legend(conds_name(conds_use), "Location", "northwest");
set(gca, 'FontSize', 12);
set(gcf, 'color', 'w');
set(gca, 'LineWidth', 1);
set(gca, 'Box', 'on');

saveas(gcf,'../all_figs/Figure_10a.svg');

% Estimate thresholds at d' = 1 using linear interpolation.
threshold_d_prime = 1;
num_conditions = size(d_prime_all, 2);
num_bootstrap = 10000;
thresholds = zeros(num_conditions, 1);
threshold_sem = zeros(num_conditions, 1);

% Create a finer grid of damping ratios for interpolation
damps_fine = linspace(min(damps_cond(2:end)), max(damps_cond(2:end)), 1000);

% Calculate mean thresholds using d_prime_all
for i = 1:num_conditions
    % Get the d_prime values for the current condition
    condition_d_prime = d_prime_all(:, i);
    
    % Perform linear interpolation.
    d_prime_interp = interp1(damps_cond(2:end), condition_d_prime, damps_fine, 'linear');
    
    % Find the index where d' is closest to the threshold 
    [~, idx] = min(abs(d_prime_interp - threshold_d_prime));
    
    % Get the corresponding damping ratio
    thresholds(i) = damps_fine(idx);
    
    % Calculate SEM using bootstrap
    % Initialize array for bootstrap thresholds
    bootstrap_thresholds = zeros(num_bootstrap, 1);
    
    % For each bootstrap iteration
    for b = 1:num_bootstrap
        bootstrap_dprimes = squeeze(bootstrap_d_prime(b,i,:));
        
        % Interpolate bootstrap d' values
        bootstrap_dprimes_interp = interp1(damps_cond(2:end), bootstrap_dprimes', damps_fine, 'linear');
        
        % Find threshold for this bootstrap sample
        [~, idx] = min(abs(bootstrap_dprimes_interp - threshold_d_prime));
        bootstrap_thresholds(b) = damps_fine(idx);
    end
    
    % Calculate SEM from bootstrap samples
    threshold_sem(i) = std(bootstrap_thresholds)/2;
end

% Plot thresholds with SEM error bars
figure(3)
set(gcf, 'Units', 'inches', 'Position', [1 1 5 5])

bar_h = bar(thresholds, 'FaceColor', [0.7 0.7 0.7]);  % Set bars to gray color
hold on
errorbar(1:num_conditions, thresholds, threshold_sem, 'k', 'LineStyle', 'none', 'CapSize', 0, "LineWidth", 4);
xlabel('Condition')
ylabel('Damping ratio threshold')
ylim([0.3, 0.9])
set(gca, 'XTickLabel', conds_name(conds_use))
xtickangle(45)

set(gca, 'FontSize', 12)
set(gcf, 'color', 'w')
set(gca, 'LineWidth', 1)
set(gca, 'Box', 'off')

saveas(gcf,'../all_figs/Figure_10c.svg');

% Calculate bootstrap standard errors for d_prime_all
n_bootstrap = 10000;
bootstrap_d_prime = zeros(n_bootstrap, size(d_prime_all, 1), size(d_prime_all, 2));

for b = 1:n_bootstrap
    % Bootstrap resample participants
    bootstrap_indices = randsample(size(hit_rate_all_part, 1), size(hit_rate_all_part, 1), true);
    
    % Calculate mean hit rates and false positive rates for this bootstrap sample
    bootstrap_hit_rates = mean(hit_rate_all_part(bootstrap_indices, :, :), 1);
    bootstrap_fp_rates = mean(fp_rate_all_part(bootstrap_indices, :), 1);
    
    % Apply the same corrections as before
    bootstrap_hit_rates(bootstrap_hit_rates == 1) = 0.95;
    bootstrap_hit_rates(bootstrap_hit_rates == 0) = 0.05;
    bootstrap_fp_rates(bootstrap_fp_rates == 1) = 0.95;
    bootstrap_fp_rates(bootstrap_fp_rates == 0) = 0.05;
    
    % Calculate d' for this bootstrap sample
    bootstrap_d_prime(b, :, :) = norminv(squeeze(bootstrap_hit_rates))' - norminv(bootstrap_fp_rates);
end

% Apply the same maximum d' constraint
bootstrap_d_prime(bootstrap_d_prime > dp_max) = dp_max;

% Calculate standard error from bootstrap samples
std_dprime_all = squeeze(std(bootstrap_d_prime, 0, 1));

        load("likelihood_values.mat")

likelihood_values = mean(log_likelihood_means,[2 3]);
likelihood_stds = avg_std_vars;
        
% Define explicit colors for all conditions
condition_colors = [
    0.0000, 0.4470, 0.7410;  % blue
    0.8500, 0.3250, 0.0980;  % orange
    0.9290, 0.6940, 0.1250;  % yellow
    0.4940, 0.1840, 0.5560;  % purple
    0.4660, 0.6740, 0.1880;  % green
    0.3010, 0.7450, 0.9330   % light blue
];

% Create new figure for threshold vs likelihood plot
figure(4)
set(gcf, 'Units', 'inches')
set(gcf, 'Position', [0, 0, 5, 4])

% Plot thresholds vs likelihood values with matching colors
for i = 1:length(thresholds)-1
    scatter(likelihood_values(i), thresholds(i), 100, 'filled', 'MarkerFaceColor', condition_colors(i,:))
    hold on
    
    % Vertical error bars (threshold)
    line([likelihood_values(i) likelihood_values(i)], ...
         [thresholds(i)-threshold_sem(i) thresholds(i)+threshold_sem(i)], ...
         'Color', condition_colors(i,:), 'LineWidth', 2)
    
    % Horizontal error bars (likelihood)
    line([likelihood_values(i)-likelihood_stds(i) likelihood_values(i)+likelihood_stds(i)], ...
         [thresholds(i) thresholds(i)], ...
         'Color', condition_colors(i,:), 'LineWidth', 2)
end
hold off

% Add labels and formatting
xlabel('Log likelihood')
ylabel('Damping ratio threshold')
ylim([0.3, 0.9])

set(gcf, 'color', 'w')
set(gca, 'LineWidth', 1)
set(gca, 'Box', 'on')

saveas(gcf,'../all_figs/Figure_10d.svg');

