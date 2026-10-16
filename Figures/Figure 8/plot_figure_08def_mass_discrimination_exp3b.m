clearvars; close all;
cond_names = ["Recorded","Full generative model","Inverted covariances","Scaled variances","Uniform model","Inverted distribution","Linear decay"];

load("Exp3b_data.mat")

    for size_p = 1:5
        total_res_cond = squeeze(nanmean(resp_cond(:,:,size_p,:),[1 4]));
        partwise_mean = squeeze(nanmean(resp_cond(:,:,size_p,:),[4]));
        partwise_data(:,:,size_p) = partwise_mean;
        cond_std = nanstd(partwise_mean,[],[1])/sqrt(size(resp_cond,1));
        tot_res_mean(size_p,:) = total_res_cond;
        tot_res_std(size_p,:) = cond_std;

    end

        for condss= [1 2 3 4 5 6 7]
            figure(2)
            % set the size of the figure so it is readble in the paper
            set(gcf, 'Units', 'inches')
            set(gcf, 'Position', [1, 1, 5, 3])
            errorbar((1:5)+0.05*randn,tot_res_mean(:,condss),tot_res_std(:,condss),'LineWidth',2,'CapSize',0)
            hold on
        end
        set(gca,'FontSize',12)
        set(gca,'XTick',[1:5])
        set(gca,'XTickLabel',{'1.8','3.16','10','31.6','100'})
        xlabel('Mass ratio')
        ylabel('Performance')
        title("Mass discrimination experiment (n = "+ num2str(size(resp_cond,1))+")")
        ylim([0.45 1])
        hold on
        % draw a dashed line to show chance performance
        plot([0 5],[0.5 0.5],'--','LineWidth',2,'Color','k')
        hold off
       
        saveas(gcf,'../all_figs/Figure_8d.svg');

        mass_ratios = [1.8, 3.16, 10, 31.6, 100]; % Define the mass ratios

        target_performance = 0.8; 

        mass_ratios_at_08 = zeros(1, size(tot_res_mean, 2)); % Pre-allocate an array to store the results

        for cond = 1:size(tot_res_mean, 2)
            % Find the first crossing of the target performance.
            % Find first crossing of target performance
            perf = tot_res_mean(:, cond);
            idx = find(perf >= target_performance, 1, 'first');
            if isempty(idx)
                mass_ratios_at_08(cond) = 100;
            elseif idx == 1
                mass_ratios_at_08(cond) = 1.8;
            else
                % Interpolate between points where performance crosses target
                x1 = mass_ratios(idx-1);
                x2 = mass_ratios(idx);
                y1 = perf(idx-1);
                y2 = perf(idx);
                mass_ratios_at_08(cond) = x1 + (target_performance - y1) * (x2 - x1) / (y2 - y1);
            end
            
        end

        disp("Mass ratios that lead to a performance of 0.8 for each condition:")
        disp(mass_ratios_at_08)


        % Pre-allocate to store results (participants x conditions x sizes)
        mass_ratios_at_08_individual = zeros(size(partwise_data, 1), size(partwise_data, 2)); 

        for part_ind = 1:size(partwise_data, 1) % Loop through each participant
            for cond = 1:size(partwise_data, 2) % Loop through each condition
                    
                    % Get individual performance across mass ratios for this condition and size
                    individual_performance = squeeze(partwise_data(part_ind, cond, :)) +0.000001*(1:5)'; 
                    
                    % Interpolate to find the mass ratio where performance is closest to 0.8
                    mass_ratios_at_08_individual(part_ind, cond) = interp1(individual_performance, mass_ratios, target_performance, 'linear', 'extrap');
                
            end
        end

        % Clip individual thresholds to the supported range.
        mass_ratios_at_08_individual(mass_ratios_at_08_individual < 1.8) = 1.8;
        mass_ratios_at_08_individual(mass_ratios_at_08_individual > 100) = 100;

        mass_ratios_at_08_individual_std = std(mass_ratios_at_08_individual,[],1,"omitmissing");
        disp("Mass ratios that lead to a performance of 0.8 for each participant, condition, and size:")
        disp(mass_ratios_at_08_individual_std) 

        % Number of bootstrap samples
        n_bootstraps = 10000;

        % Pre-allocate to store bootstrapped mass ratios
        bootstrapped_mass_ratios = zeros(n_bootstraps, length(mass_ratios_at_08));

        % Perform bootstrapping
        for b = 1:n_bootstraps
            % Randomly sample participants with replacement
            sampled_participants = randi(size(partwise_data, 1), [randi(size(partwise_data, 1)), 1]);
            
            % Calculate mean performance for sampled participants
            sampled_performance = zeros(5, size(partwise_data, 2));
            for cond = 1:size(partwise_data, 2)
                % Average across sampled participants for each mass ratio and condition
                sampled_performance(:, cond) = mean(partwise_data(sampled_participants, cond, :), 1);
            end
            
            % Find threshold (mass ratio at target performance) for each condition
            for cond = 1:size(partwise_data, 2)
                perf = sampled_performance(:, cond);
                idx = find(perf > target_performance, 1, 'first');
                
                if isempty(idx)
                    bootstrapped_mass_ratios(b, cond) = NaN;
                elseif idx==1
                        x1 = mass_ratios(idx);
                    x2 = mass_ratios(idx+1);
                    y1 = perf(idx);
                    y2 = perf(idx+1);
                    bootstrapped_mass_ratios(b, cond) = x1 + (target_performance - y1) * (x2 - x1) / (y2 - y1);
                else
                   % Interpolate between points where performance crosses target
                    x1 = mass_ratios(idx-1);
                    x2 = mass_ratios(idx);
                    y1 = perf(idx-1);
                    y2 = perf(idx);
                    bootstrapped_mass_ratios(b, cond) = x1 + (target_performance - y1) * (x2 - x1) / (y2 - y1);
                end
            end
        end
        % Clip bootstrapped mass ratios between 1.8 and 100
        bootstrapped_mass_ratios(bootstrapped_mass_ratios < 1.8) = 1.8;
        bootstrapped_mass_ratios(bootstrapped_mass_ratios > 100) = 100;
        % Calculate the standard deviation of the bootstrapped mass ratios in log scale
        bootstrapped_std = std(log(bootstrapped_mass_ratios), 'omitnan');

        % Plot the average mass ratios at 0.8 for each condition
        figure(3)
        % set the size of the figure so it is readable in the paper
        set(gcf, 'Units', 'inches')
        set(gcf, 'Position', [1, 1, 5, 3])
        b = bar(mass_ratios_at_08);
        b.FaceColor = [0.5 0.5 0.5]; % Set bar color to gray
        hold on
        
        upper_bound = exp(log(mass_ratios_at_08) + bootstrapped_std);
        lower_bound = exp(log(mass_ratios_at_08) - bootstrapped_std);
        
        % Plot error bars using the asymmetric bounds
        for i = 1:length(mass_ratios_at_08)
            line([i i], [lower_bound(i) upper_bound(i)], 'Color', 'k', 'LineWidth', 3)
        end

        set(gca,'FontSize',12)
        set(gca,'XTick',[1:7])
        set(gca,'XTickLabel',cond_names)
        xlabel('Condition')
        ylabel('Mass ratio')
        set(gca, 'YScale', 'log') % Set y-axis to logarithmic scale
        yticks([1.8, 3.16, 10, 31.6, 100]) % Set y-axis ticks to mass ratio values
        yticklabels({'1.8', '3.16', '10', '31.6', '100'}) % Label y-axis ticks
        ylim([1, 130]) % Set y-axis limits
        saveas(gcf,'../all_figs/Figure_8e.svg');
        hold off

        load("likelihood_values.mat")

likelihood_values = mean(log_likelihood_means,[2 3]);
likelihood_stds = avg_std_vars;
        
% Create new figure for threshold vs likelihood plot
figure(4)
set(gcf, 'Units', 'inches')
set(gcf, 'Position', [1, 1, 5, 3])
set(gca,'Box','on')

% Get default MATLAB colors
colors = get(gca,'ColorOrder');

% Plot thresholds vs likelihood values with matching colors
for i = 1:6
    scatter(likelihood_values(i), mass_ratios_at_08(i), 100, colors(i,:), 'filled')
    hold on
    
    % Add error bars with matching colors
    line([likelihood_values(i) likelihood_values(i)], ...
         [lower_bound(i) upper_bound(i)], 'Color', colors(i,:), 'LineWidth', 2)
    
    line([likelihood_values(i)-likelihood_stds(i) likelihood_values(i)+likelihood_stds(i)], ...
         [mass_ratios_at_08(i) mass_ratios_at_08(i)], 'Color', colors(i,:), 'LineWidth', 2)
end

set(gca, 'YScale', 'log')
set(gca,'FontSize',12)
xlabel('Log likelihood')
ylabel('Mass ratio threshold')
yticks([1.8, 3.16, 10, 31.6, 100])
yticklabels({'1.8', '3.16', '10', '31.6', '100'})
ylim([1, 130])
set(gca,'Box','on')

% Optional: Save the figure
saveas(gcf,'../all_figs/Figure_8f.svg');
        