clearvars; close all;
load("Expt2_data.mat")
part_ind1=1;
titles = ["Recorded sounds", "Full generative model", "Inverted covariances","Scaled variances","2 modes Gaussian","Uniform model","Inverted distributions","Linear decay"];
for part_ind1=1:size(resp_fin,1)

    resp_part = resp_fin(part_ind1,:);

    for cond = 1:8
        % get the confusion matrix for each condition
        C{part_ind1,cond} = confusionmat(matnum_part(key_part==cond),resp_part(key_part==cond));
        if part_ind1==1
            C1{cond} = zeros(7,7);
        end
        C1{cond} = C1{cond}+confusionmat(matnum_part(key_part==cond),resp_part(key_part==cond));
    end
    part_ind1 = part_ind1+1;
    end

%plot the confusion matrices averaged across participants
figure;
% Create custom white-to-blue colormap
cmap = [ones(64,1), ones(64,1), ones(64,1)]; % Start with white
blue_gradient = linspace(1, 0, 64)';  % Gradually decrease R and G to create blue
cmap(:,1) = blue_gradient;
cmap(:,2) = blue_gradient;

for cond = 1:8
    subplot(3,3,cond)
    imagesc(C1{cond}./sum(C1{cond}(:,:),2))
    title(titles(cond))
    colormap(cmap)  % Apply custom colormap
    clim([0 0.5])
    axis square
    % Only show labels and colorbar for the first subplot (condition 2)
    if cond == 2
        xticks(1:7)
        yticks(1:7)
        yticklabels({'metal','glass','ceramic','stone','wood','plastic','cardboard'})
        xticklabels({'metal','glass','ceramic','stone','wood','plastic','cardboard'})
        colorbar
    else
        xticks([])
        yticks([])
    end
end

% Get the percent correct from the confusion matrices and plot it
for part_ind = 1:(size(C,1))
for cond = 1:8
    percent_corr(cond) = sum(diag(C1{cond}))/sum(C1{cond}(:));
    percent_corr_part(part_ind,cond) = sum(diag([C{part_ind,cond}]))/sum([C{part_ind,cond}],'all');
end
end
std_corr = std(percent_corr_part,0,1);
mean_percent_corr= mean(percent_corr_part,1);

% Calculate the correlation of the confusion matrices for condition 2 to 9 with condition 1 and also calculate the std across participants
for cond = 2:8
    corr_cond(cond-1) = corr(C1{1}(:),C1{cond}(:));
    for ii = 1:(part_ind1-1)
        % Find correlation of confusion matrices for each participant
        corr_cond_part(ii,cond-1) = corr(C{ii,1}(:),C{ii,cond}(:));
    end
    % Find std across participants
    std_cond(cond-1) = std(corr_cond_part(:,cond-1))/sqrt(size(resp_fin,1));
end

% Calculate the correlation of the confusion matrices for all other conditions with condition 2
n_bootstrap = 10000; % Number of bootstrap samples
for cond = 2:8
    corr_cond2(cond-1) = corr(C1{1}(:),C1{cond}(:));
    
    % Initialize bootstrap correlations array
    bootstrap_corrs = zeros(n_bootstrap, 1);
    
    % Perform bootstrap
    for b = 1:n_bootstrap
        % Random sampling with replacement
        boot_indices = randi([1, part_ind1-1], part_ind1-1, 1);
        
        % Calculate average confusion matrix for bootstrapped sample
        boot_matrix_cond2 = zeros(7,7);
        boot_matrix_cond = zeros(7,7);
        
        for idx = 1:length(boot_indices)
            boot_matrix_cond2 = boot_matrix_cond2 + C{boot_indices(idx),1};
            boot_matrix_cond = boot_matrix_cond + C{boot_indices(idx),cond};
        end
        
        boot_matrix_cond2 = boot_matrix_cond2 / length(boot_indices);
        boot_matrix_cond = boot_matrix_cond / length(boot_indices);
        
        % Calculate correlation for this bootstrap sample
        bootstrap_corrs(b) = corr(boot_matrix_cond2(:), boot_matrix_cond(:));
    end
    
    % Calculate standard error from bootstrap distribution
    std_cond2(cond-1) = std(bootstrap_corrs);
end
figure;
bar(corr_cond2, 'FaceColor', [0.5 0.5 0.5])  % Set bars to gray
hold on
errorbar(corr_cond2,std_cond2,'k.',"LineWidth",3,"CapSize",0)
xlabel('Condition')
ylabel('Correlation')
set(gca,'XTickLabel',titles(2:end))
%Font size
set(gca,'FontSize',18)

% Total all elements in the first (4,4) submatrix of the confusion matrices and the last (3,3) submatrix, add them together and divide by the total number of elements in the confusion matrices
for cond = 1:8
    percent_corr2(cond) = (sum(sum(C1{cond}(1:4,1:4)))+sum(sum(C1{cond}(5:7,5:7))))/sum(C1{cond}(:));
end

saveas(1,"../all_figs/Figure_7b.svg")
saveas(2,"../all_figs/Figure_7c.svg")

% Calculate proportion correct for Recorded condition (condition 2)
recorded_correct = sum(diag(C1{2}))/sum(C1{2}(:));

% Calculate proportion correct for each participant
for ii = 1:(part_ind1-1)
    recorded_correct_part(ii) = sum(diag(C{ii,2}))/sum(C{ii,2}(:));
end

% Calculate standard error
recorded_std = std(recorded_correct_part);

disp(['Proportion correct for Recorded condition (Exp2): ' num2str(recorded_correct) ' ± ' num2str(recorded_std)])

