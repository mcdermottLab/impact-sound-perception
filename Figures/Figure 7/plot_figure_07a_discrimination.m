clearvars; close all;
load("results_exp1.mat")

condnames = ["Full generative model","Inverted covariances","2 modes Gaussian","Scaled variances","Uniform model","Inverted Distributions","Linear decay"];

total_res_cond = min(nanmean(resp_cond,[1 3]), 1);
partwise_mean = min(squeeze(mean(resp_cond,[3])), 1);
cond_std = nanstd(partwise_mean,[],[1])/sqrt(size(resp_cond,1));

figure;
set(gcf,'Units','inches','Position',[0 0 6 4])
bar(total_res_cond,"FaceColor",[0.5 0.5 0.5])
set(gca,'FontSize',14)
set(gca,'XTick',[1:7])
set(gca,'XTickLabel',condnames)
xlabel('Condition')
ylabel('Proportion Correct')
ylim([0.3 1])
hold on
errorbar(total_res_cond,cond_std,'.','LineWidth',3,'Color','k','CapSize',0)
% draw a dashed line to show chance performance
plot([0 9],[0.5 0.5],'--','LineWidth',2,'Color','k')
xlim([0 8])

hold off

saveas(1,"../all_figs/Figure_7a.svg")

% Calculate the overall standard deviation for the difference across conditions 1 and 2
cond1_data = partwise_mean(:, 1);
cond2_data = partwise_mean(:, 2);

% Calculate the difference between the two conditions
diff_data = cond1_data - cond2_data;

% Calculate the overall standard deviation of the difference
overall_sd_diff = std(diff_data);

% Display the result
fprintf('Overall Standard Deviation for the difference between conditions 1 and 2: %.3f\n', overall_sd_diff);

%% Power Analysis for t-tests against condition 1

% Initialize variables to store results
num_conditions = 7;
effect_sizes = zeros(num_conditions-1, 1);
sample_sizes = zeros(num_conditions-1, 1);
powers = zeros(num_conditions-1, 1);
p_vals = zeros(num_conditions-1, 1);

% Reference condition (condition 1)
data1 = partwise_mean(:,1);
data1 = data1(:);

% Loop through each condition (comparing with condition 1)
for i = 2:num_conditions
    % Data for the comparison condition
    data2 = partwise_mean(:,i);
    data2 = data2(:);

    % Perform t-test
    [~, p, ~, stats] = ttest(data1, data2);
    
    % Store p-value
    p_vals(i-1) = p;

    % Calculate effect size (Cohen's d)
    effect_sizes(i-1) = abs(stats.tstat) / sqrt(size(resp_cond, 1));

    % Power analysis to find sample size for 90% power
    sample_sizes(i-1) = sampsizepwr('t2', [size(resp_cond, 1), size(resp_cond, 1)], effect_sizes(i-1), 0.9);
end

% Compare condition 1 with chance performance (0.5).
[h, p_chance, ci, stats_chance] = ttest(data1, 0.5);
fprintf('\nT-test results for Condition 1 vs Chance (0.5):\n');
fprintf('t(%d) = %.3f, p = %.4f\n', stats_chance.df, stats_chance.tstat, p_chance);
fprintf('95%% CI: [%.3f, %.3f]\n', ci(1), ci(2));