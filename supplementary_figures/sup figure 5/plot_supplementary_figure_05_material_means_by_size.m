close all; clearvars

material = ["glass","ceramic","metal","stone","wood","plastic","cardboard"];
size_buckets = 1:4;
load_dir = "../../data/modes_plot/";

% Bootstrap parameters
n_bootstrap = 10000;  % number of bootstrap samples
bootstrap_alpha = 0.3;  % transparency for contours
colorscheme = lines(length(material));

close all

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

% Initialize arrays for per-material means and standard errors
material_mean_freq = zeros(length(material), length(size_buckets));
material_mean_amp = zeros(length(material), length(size_buckets));
material_mean_rt60 = zeros(length(material), length(size_buckets));
material_se_freq = zeros(length(material), length(size_buckets));
material_se_amp = zeros(length(material), length(size_buckets));
material_se_rt60 = zeros(length(material), length(size_buckets));

% Calculate means and SEs per material
for ii = 1:length(material)
    for jjj = 1:length(size_buckets)
        size_bucket = size_buckets(jjj);
        load(load_dir+material(ii)+"_"+string(size_bucket)+"_modes.mat")
        
        % Get the squeezed data
        freq_data = squeeze(modes_new(1,:,:));
        amp_data = squeeze(modes_new(2,:,:));
        rt60_data = squeeze(modes_new(3,:,:));
        
        % Calculate mean across modes for each IR
        ir_means_freq = mean(freq_data, 2, 'omitnan');
        ir_means_amp = mean(amp_data, 2, 'omitnan');
        ir_means_rt60 = mean(rt60_data, 2, 'omitnan');
        
        % Remove any NaN IRs
        valid_ir_freq = ir_means_freq(~isnan(ir_means_freq));
        valid_ir_amp = ir_means_amp(~isnan(ir_means_amp));
        valid_ir_rt60 = ir_means_rt60(~isnan(ir_means_rt60));
        
        % Bootstrap calculations
        boot_means_freq = zeros(1, n_bootstrap);
        boot_means_amp = zeros(1, n_bootstrap);
        boot_means_rt60 = zeros(1, n_bootstrap);
        
        for b = 1:n_bootstrap
            boot_idx = randi(length(valid_ir_freq), [1, length(valid_ir_freq)]);
            boot_means_freq(b) = mean(valid_ir_freq(boot_idx));
            boot_means_amp(b) = mean(valid_ir_amp(boot_idx));
            boot_means_rt60(b) = mean(valid_ir_rt60(boot_idx));
        end
        
        % Store means and standard errors
        material_mean_freq(ii,jjj) = mean(valid_ir_freq);
        material_mean_amp(ii,jjj) = mean(valid_ir_amp);
        material_mean_rt60(ii,jjj) = mean(valid_ir_rt60);
        
        material_se_freq(ii,jjj) = std(boot_means_freq);
        material_se_amp(ii,jjj) = std(boot_means_amp);
        material_se_rt60(ii,jjj) = std(boot_means_rt60);
    end
end

% Plot for Frequency (with legend)
figure(1)
set(gcf,'Units','inches','Position',[1,1,3,4])
hold on
for ii = 1:length(material)
    errorbar(1:length(size_buckets), material_mean_freq(ii,:), material_se_freq(ii,:), '-o', ...
        'LineWidth', 2, 'Color', colorscheme(ii,:), 'MarkerFaceColor', colorscheme(ii,:), 'CapSize', 0)
end
  % Keep legend only for frequency plot
ylim([0, 10000])

% Plot for Amplitude (no legend)
figure(2)
set(gcf,'Units','inches','Position',[1,1,3,4])
hold on
for ii = 1:length(material)
    errorbar(1:length(size_buckets), material_mean_amp(ii,:), material_se_amp(ii,:), '-o', ...
        'LineWidth', 2, 'Color', colorscheme(ii,:), 'MarkerFaceColor', colorscheme(ii,:), 'CapSize', 0)
end
legend(material, 'Location', 'best')
ylim([-40, 0])

% Plot for RT60 (no legend)
figure(3)
set(gcf,'Units','inches','Position',[1,1,3,4])
hold on
for ii = 1:length(material)
    errorbar(1:length(size_buckets), material_mean_rt60(ii,:), material_se_rt60(ii,:), '-o', ...
        'LineWidth', 2, 'Color', colorscheme(ii,:), 'MarkerFaceColor', colorscheme(ii,:), 'CapSize', 0)
end
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
        ylabel('Frequency (Hz)')
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
    save_dir = load_dir+"figs_mat_means_v3/";
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    if fig == 1
        saveas(gcf, "../../Figures/all_figs/SupFigure5_4.svg")
    elseif fig == 2
        saveas(gcf, "../../Figures/all_figs/SupFigure5_5.svg")
    else
        saveas(gcf, "../../Figures/all_figs/SupFigure5_6.svg")
    end
end
