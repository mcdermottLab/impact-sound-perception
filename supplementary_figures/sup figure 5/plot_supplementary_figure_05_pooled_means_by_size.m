close all; clearvars

material = ["wood","glass","metal","cardboard","plastic","ceramic","stone"];
size_buckets = 1:4;
load_dir = "../../data/modes_plot/";

% Bootstrap parameters
n_bootstrap = 10000;  % number of bootstrap samples
bootstrap_alpha = 0.3;  % transparency for contours
colorscheme = lines(length(material));

% Define size labels at the start
size_labels = ["smallest", "small", "medium", "large"];

% Create figures for each mode parameter
figure(1)  % Frequency
figure(2)  % Amplitude
figure(3)  % RT60

% Initialize arrays to store all modes for each size bucket
all_freq_by_size = cell(1, length(size_buckets));
all_amp_by_size = cell(1, length(size_buckets));
all_rt60_by_size = cell(1, length(size_buckets));

% Initialize each cell with an empty matrix
for jjj = 1:length(size_buckets)
    all_freq_by_size{jjj} = [];
    all_amp_by_size{jjj} = [];
    all_rt60_by_size{jjj} = [];
end

% Collect all modes for each size bucket across materials
for ii = 1:length(material)
    for jjj = 1:length(size_buckets)
        size_bucket = size_buckets(jjj);
        load(load_dir+material(ii)+"_"+string(size_bucket)+"_modes.mat")
        
        % Get the squeezed data
        freq_data = squeeze(modes_new(1,:,:));
        amp_data = squeeze(modes_new(2,:,:));
        rt60_data = squeeze(modes_new(3,:,:));
        
        % Concatenate impulse responses across materials.
        if isempty(all_freq_by_size{jjj})
            all_freq_by_size{jjj} = freq_data;
            all_amp_by_size{jjj} = amp_data;
            all_rt60_by_size{jjj} = rt60_data;
        else
            all_freq_by_size{jjj} = [all_freq_by_size{jjj}; freq_data];  % Vertical concatenation
            all_amp_by_size{jjj} = [all_amp_by_size{jjj}; amp_data];
            all_rt60_by_size{jjj} = [all_rt60_by_size{jjj}; rt60_data];
        end
    end
end

% Calculate bootstrap means and confidence intervals
overall_mean_freq = zeros(1, length(size_buckets));
overall_mean_amp = zeros(1, length(size_buckets));
overall_mean_rt60 = zeros(1, length(size_buckets));
overall_se_freq = zeros(1, length(size_buckets));
overall_se_amp = zeros(1, length(size_buckets));
overall_se_rt60 = zeros(1, length(size_buckets));

for jjj = 1:length(size_buckets)
    % Calculate mean across modes for each IR
    ir_means_freq = mean(all_freq_by_size{jjj}, 2, 'omitnan');  % mean across modes (dim 2) for each IR
    ir_means_amp = mean(all_amp_by_size{jjj}, 2, 'omitnan');
    ir_means_rt60 = mean(all_rt60_by_size{jjj}, 2, 'omitnan');
    
    % Remove any NaN IRs
    valid_ir_freq = ir_means_freq(~isnan(ir_means_freq));
    valid_ir_amp = ir_means_amp(~isnan(ir_means_amp));
    valid_ir_rt60 = ir_means_rt60(~isnan(ir_means_rt60));
    
    % Bootstrap calculations across IRs
    boot_means_freq = zeros(1, n_bootstrap);
    boot_means_amp = zeros(1, n_bootstrap);
    boot_means_rt60 = zeros(1, n_bootstrap);
    
    for b = 1:n_bootstrap
        % Bootstrap sampling of IRs
        boot_idx = randi(length(valid_ir_freq), [1, length(valid_ir_freq)]);
        boot_means_freq(b) = mean(valid_ir_freq(boot_idx));
        boot_means_amp(b) = mean(valid_ir_amp(boot_idx));
        boot_means_rt60(b) = mean(valid_ir_rt60(boot_idx));
    end
    
    % Calculate overall means and standard errors
    overall_mean_freq(jjj) = mean(valid_ir_freq);
    overall_mean_amp(jjj) = mean(valid_ir_amp);
    overall_mean_rt60(jjj) = mean(valid_ir_rt60);
    
    overall_se_freq(jjj) = std(boot_means_freq);
    overall_se_amp(jjj) = std(boot_means_amp);
    overall_se_rt60(jjj) = std(boot_means_rt60);
end

% Plot for Frequency
figure(1)
set(gcf,'Units','inches','Position',[1,1,3,4])
errorbar(1:length(size_buckets), overall_mean_freq/1000, overall_se_freq/1000, '-o', 'LineWidth', 2, 'Color', 'k', 'MarkerFaceColor', 'k', 'CapSize', 0)
ylim([0, 10])

% Plot for Amplitude
figure(2)
set(gcf,'Units','inches','Position',[1,1,3,4])
errorbar(1:length(size_buckets), overall_mean_amp, overall_se_amp, '-o', 'LineWidth', 2, 'Color', 'k', 'MarkerFaceColor', 'k', 'CapSize', 0)
ylim([-40, 0])
% Plot for RT60
figure(3)
set(gcf,'Units','inches','Position',[1,1,3,4])
errorbar(1:length(size_buckets), overall_mean_rt60, overall_se_rt60, '-o', 'LineWidth', 2, 'Color', 'k', 'MarkerFaceColor', 'k', 'CapSize', 0)
ylim([0, 0.2])
% Format figures
for fig = 1:3
    figure(fig)
    xlabel('Size Category')
    xticks(1:length(size_buckets))
    xticklabels(size_labels)
    xlim([0.5, length(size_buckets)+0.5])
    
    % Set appropriate y-label and title for each figure
    if fig == 1
        ylabel('Frequency (kHz)')
        title('Mean Frequency vs Size')
    elseif fig == 2
        ylabel('Amplitude (dB)')
        title('Mean Amplitude vs Size')
    else
        ylabel('RT60 (s)')
        title('Mean RT60 vs Size')
    end
    
    % Apply common formatting
    set(findall(gcf,'-property','FontSize'),'FontSize',12)
    set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
    set(findall(gcf,'-property','LineWidth'),'LineWidth',2)
    set(findall(gcf,'-property','MarkerSize'),'MarkerSize',6)
    set(gca,'Box','on')
    set(gca,'XColor','k')
    set(gca,'YColor','k')
    
    % Save figures
    save_dir = load_dir+"figs_mat_means_v2/";
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    if fig == 1
        saveas(gcf, "../../Figures/all_figs/SupFigure5_1.svg")
    elseif fig == 2
        saveas(gcf, "../../Figures/all_figs/SupFigure5_2.svg")
    else
        saveas(gcf, "../../Figures/all_figs/SupFigure5_3.svg")
    end
end
