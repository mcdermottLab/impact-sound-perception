clearvars; close all;
load("model_result_200.mat")

damps_cond = [0,0.25,0.5,0.75,0.9]; 
conds_name = ["Recorded","Full generative model","Inverted covariance","Scaled Variance","Uniform model","Inverted distribution","Linear decay"];

mean_perf = squeeze(mean(mod_ans_cond,1));
std_perf = squeeze(std(mod_ans_cond,0,1))/sqrt(size(mod_ans_cond,1));
eps = 0.001;

conds_use = [1 2 3 4 5];

for condss= conds_use
figure(1)
% set the size of the figure so it is readble in the paper
set(gcf,'Units','Inches','Position',[1, 1, 6, 5])
errorbar(damps_cond(1:5),mean_perf(:,condss),std_perf(:,condss),'-o','LineWidth',3,'CapSize',0,'MarkerSize',5)
hold on
end
set(gca,'FontSize',14)
set(gca,'XTick',damps_cond)
xlabel('Damping ratio')
ylabel('Proportion correct')
title("Damping discrimination experiment (n = "+ num2str(size(mod_ans_cond,1))+")")
ylim([0.45 1])
xlim([-0.1 1])
hold on
% draw a dashed line to show chance performance
plot([-1 1.2],[0 0],'--','LineWidth',2,'Color','k')
hold off
legend(conds_name(conds_use),'Location','southeast','Box','off')
saveas(1,"../../Figures/all_figs/SupFigure_8.svg")

