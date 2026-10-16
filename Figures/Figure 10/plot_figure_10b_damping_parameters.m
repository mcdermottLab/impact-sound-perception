close all; clearvars;

% Add extra_scripts to path for csv2cell function
addpath('../../extra_scripts');

loaddir = 'dampIRs_test3';
% Load csv files
Amp = csvread(fullfile(loaddir, 'Amp_1.csv'));
RT60 = csvread(fullfile(loaddir, 'RT60_1.csv'));
freq = csvread(fullfile(loaddir, 'freq_1.csv'));
alpha = csvread(fullfile(loaddir, 'alpha_1.csv'));
gamma = csvread(fullfile(loaddir, 'gamma_1.csv'));

directories = readmatrix(fullfile(loaddir, 'directories.csv'));
[~,dir_order] = sort(directories(:,3));

dirnames = csv2cell(loaddir+"/dirnames.csv",'fromfile');

% Extract the string between the last two /'s in the dirnames
for i = 1:length(dirnames)
    dirnames1_tmp = extractBetween(dirnames{i},'/','/');
    dirnames1{i} = dirnames1_tmp{end};
    % Extract everything before the first _ as material and everything after last _ and before percent as object name
    object_materials{i} = extractBefore(dirnames1{i},'_');
    object_names_tmp = extractBetween(dirnames1{i},'_','percent');
    % Separate object_name_tmp at "_"
    object_names_tmp = split(object_names_tmp,'_');
    damping_cond{i} = object_names_tmp{end};

end

all_mats = unique(object_materials);
damps_cond = ["00","25","50","75"];
indd =0;

RT60_1 = nan(length(all_mats),length(damps_cond),size(RT60,2)*3);
freq_1 = nan(length(all_mats),length(damps_cond),size(RT60,2)*3);

for indd = 1: length(all_mats)
    mat = all_mats{indd};
    for indd2 = 1:length(damps_cond)
        damp = damps_cond{indd2};
        % Find the indices of the current material
        ind = find(contains(object_materials,mat) & contains(damping_cond,damp));
        if length(ind) ==6
            ind = ind(1:3);
        end
        RT60_tmp = RT60(dir_order(ind),:);
        freq_tmp = freq(dir_order(ind),:);
        leg_str = ["00%", "25%", "50%","75%"];
        RT60_1(indd,indd2,:) = RT60_tmp(:);
        freq_1(indd,indd2,:) = freq_tmp(:);
    end
    % Find the indices of the current material

end

object_num = 2;
RT60_1_all = squeeze(RT60_1(object_num,1,:));
RT60_2_all = squeeze(RT60_1(object_num,2,:));
RT60_3_all = squeeze(RT60_1(object_num,3,:));
RT60_4_all = squeeze(RT60_1(object_num,4,:));
freq_1_all = squeeze(freq_1(object_num,1,:));
freq_2_all = squeeze(freq_1(object_num,2,:));
freq_3_all = squeeze(freq_1(object_num,3,:));
freq_4_all = squeeze(freq_1(object_num,4,:));

% Filter out RT60 points below 0.03
RT60_threshold = 0.03;
valid_1 = RT60_1_all >= RT60_threshold;
valid_2 = RT60_2_all >= RT60_threshold;
valid_3 = RT60_3_all >= RT60_threshold;
valid_4 = RT60_4_all >= RT60_threshold-0.01;

figure(1)
set(gcf, 'Units', 'inches', 'Position', [2 2 6 4])
box on
hold on
% Plot original scatter points with filtered data
h1 = plot(freq_1_all(valid_1), RT60_1_all(valid_1), 'r.', 'MarkerSize', 12);
h2 = plot(freq_2_all(valid_2), RT60_2_all(valid_2), 'b.', 'MarkerSize', 12);
h3 = plot(freq_3_all(valid_3), RT60_3_all(valid_3), 'g.', 'MarkerSize', 12);
h4 = plot(freq_4_all(valid_4), RT60_4_all(valid_4), 'k.', 'MarkerSize', 12);

set(gca,'FontSize',14)
xlim([-3000 13000])
ylim([0 0.15])
xlabel('Frequency (Hz)')
ylabel('RT60 (s)')

% Plot confidence ellipses with filtered data
colors = {'r','b','g','k'};
all_data = {[freq_1_all(valid_1), RT60_1_all(valid_1)], ...
            [freq_2_all(valid_2), RT60_2_all(valid_2)], ...
            [freq_3_all(valid_3), RT60_3_all(valid_3)], ...
            [freq_4_all(valid_4), RT60_4_all(valid_4)]};

for i = 1:4
    data = all_data{i};
    % Calculate mean and covariance
    mu = mean(data);
    covar = cov(data);
    % Get ellipse points
    [ex,ey] = get_error_ellipse(covar, mu, 0.90);
    plot(ex, ey, colors{i}, 'LineWidth', 0.5, 'LineStyle', '--');
end
legend([h1 h2 h3 h4], leg_str, 'Box', 'off', 'Location', 'southeast')
saveas(gcf, "../all_figs/Figure_10b.svg")

function [ex,ey] = get_error_ellipse(covar, mu, conf)
    % Calculate error ellipse at confidence level conf
    chi2val = chi2inv(conf,2);
    [eigvec, eigval] = eig(covar);
    
    % Get the largest eigenvalue
    [~, largest_eigval_ind] = max(diag(eigval));
    largest_eigvec = eigvec(:,largest_eigval_ind);
    
    % Get the angle of rotation
    angle = atan2(largest_eigvec(2), largest_eigvec(1));
    
    % Get the 95% confidence interval error ellipse
    chisquare_val = sqrt(chi2val);
    theta_grid = linspace(0,2*pi,100);
    phi = angle;
    X0 = mu(1);
    Y0 = mu(2);
    a = chisquare_val*sqrt(eigval(largest_eigval_ind,largest_eigval_ind));
    b = chisquare_val*sqrt(eigval(3-largest_eigval_ind,3-largest_eigval_ind));
    
    % Parametric equation of the ellipse
    ex = X0 + a*cos(theta_grid)*cos(phi) - b*sin(theta_grid)*sin(phi);
    ey = Y0 + a*cos(theta_grid)*sin(phi) + b*sin(theta_grid)*cos(phi);
end

