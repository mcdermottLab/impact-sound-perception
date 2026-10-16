close all; clearvars 

material = [ "glass", "ceramic", "metal", "stone","wood","plastic", "cardboard"];
mat_group1 = ["glass", "ceramic", "metal", "stone"];
mat_group2 = ["wood","plastic", "cardboard"];
load_dir = "../../data/modes_synth/"; 
colorscheme = jet(length(material));
size_buckets = 1:4;
set(0,'DefaultFigureVisible','on')
freq_lims = [0, 20000];
    amp_lims = [-40, 0];
    RT60_lims = [0, 0.5];

% Create 3 separate figures for different parameter combinations
parameter_pairs = {
    struct('x', 1, 'y', 2, 'xlabel', 'Frequency (kHz)', 'ylabel', 'Amplitude (dB)'),
    struct('x', 1, 'y', 3, 'xlabel', 'Frequency (kHz)', 'ylabel', 'RT60 (s)'),
    struct('x', 3, 'y', 2, 'xlabel', 'RT60 (s)', 'ylabel', 'Amplitude (dB)')
};

for plot_type = 1:3
    figure('Units', 'inches', 'Position', [1 1 12 24]); 
    t = tiledlayout(7, 4, 'Padding', 'tight');
    t.TileSpacing = 'tight';
    
    for jjj = 1:length(size_buckets)
        for mat_idx = 1:length(material)
            nexttile((mat_idx-1)*4 + jjj); hold on;
            
            % Set square aspect ratio before plotting
            pbaspect([1 1 1]);
            
            % Load and plot data for current material and size
            material_name = material(mat_idx);
            load(load_dir + material_name + "_" + string(size_buckets(jjj)) + "_modes.mat");
            
            % Filter modes
            modes_filtered = filter_modes(modes, material_name);
            
            % Plot scatter with existing styling
            amp_norm = (modes_filtered(2,:) - min(modes_filtered(2,:))) / (max(modes_filtered(2,:)) - min(modes_filtered(2,:)));
            amp_norm(isnan(amp_norm)) = 0;
            transparency = tanh(0.4 + 0.6 * amp_norm);
            transparency(transparency > 1) = 1;

            s = scatter(modes_filtered(parameter_pairs{plot_type}.x,:), ...
                       modes_filtered(parameter_pairs{plot_type}.y,:), ...
                       abs(modes_filtered(2,:) + 50).^1, ...
                       colorscheme(mat_idx,:), 'filled');
            s.AlphaData = transparency;
            s.MarkerEdgeAlpha = 'flat';
            s.MarkerFaceAlpha = 'flat';
            
            % Set axis limits and ticks based on plot type
            if plot_type == 1
                set(gca,'xlim',freq_lims);
                set(gca,'ylim',amp_lims);
                % Only show ticks for leftmost column and bottom row
                if jjj == 1 && mat_idx == length(material)
                    set(gca, 'XTick', [0:10000:20000], 'XTickLabel', {'0', '10', '20'}, 'YTick', [-40:20:0]);
                else
                    set(gca, 'XTick', [], 'YTick', []);
                end
            elseif plot_type == 2
                set(gca,'xlim',freq_lims);
                set(gca,'ylim',RT60_lims);
                if jjj == 1 && mat_idx == length(material)
                    set(gca, 'XTick', [0:10000:20000], 'XTickLabel', {'0', '10', '20'}, 'YTick', [0:0.25:0.5]);
                else
                    set(gca, 'XTick', [], 'YTick', []);
                end
            else
                set(gca,'xlim',RT60_lims);
                set(gca,'ylim',amp_lims);
                if jjj == 1 && mat_idx == length(material)
                    set(gca, 'XTick', [0:0.25:0.5], 'YTick', [-40:20:0]);
                else
                    set(gca, 'XTick', [], 'YTick', []);
                end
            end
            
            if mat_idx == length(material) && jjj == 1
                xlabel(parameter_pairs{plot_type}.xlabel);
                ylabel(parameter_pairs{plot_type}.ylabel);
            end
            
            hold off;
        end
    end
    
    % Format and save each figure
    format_plot();
    save_figure(load_dir, size_buckets, "all_materials_plot" + plot_type + ".svg");
end

function format_plot()
    set(findall(gcf,'-property','FontSize'),'FontSize',12); % Increased font size to 10
    set(findall(gcf,'-property','FontName'),'FontName','Helvetica');
    set(findall(gcf,'-property','LineWidth'),'LineWidth',2);
    set(findall(gcf,'-property','Box'),'Box','on');
    set(findall(gcf,'-property','XColor'),'XColor','k');
    set(findall(gcf,'-property','YColor'),'YColor','k');
    set(findall(gcf,'-property','ZColor'),'ZColor','k');
    set(findall(gcf,'-property','GridColor'),'GridColor','k');
    set(findall(gcf,'-property','MinorGridColor'),'MinorGridColor','k');
    set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':');
    set(findall(gcf,'-property','GridLineStyle'),'GridLineStyle',':');
    set(findall(gcf,'-property','MinorGridAlpha'),'MinorGridAlpha',0.5);
    set(findall(gcf,'-property','GridAlpha'),'GridAlpha',0.5);
    set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':');
    
end

function save_figure(load_dir, size_bucket, filename)
    save_dir = load_dir + "figs_combined_v2/";
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    saveas(gcf, save_dir + filename);
end

function modes_filtered = filter_modes(modes,material)
    % Initialize an array to keep track of the modes to keep
    keep_indices = true(1, size(modes, 2));

    if material == "cardboard" || material == "plastic"
        param1 = 0.01;
        param2 = 1;
        param3 = 0.03;
    else
        param1 = 0.02;
        param2 = 10;
        param3 = 0.08;
    end

    for i = 1:size(modes, 2)
        for j = i+1:size(modes, 2)
            % Check if the frequency difference and RT60 difference are within the thresholds
            if (abs(modes(1,i) - modes(1,j)) < (param1*modes(1,i)) && abs(modes(2,i) - modes(2,j)) < (param2) &&abs(modes(3,i) - modes(3,j)) < (param3*modes(3,i))) || (abs(modes(1,i) - modes(1,j)) < 10 && abs(modes(1,i))>8000)
                % Mark the mode with the lower amplitude for removal
                if modes(2,i) < modes(2,j)
                    keep_indices(i) = false;
                else
                    keep_indices(j) = false;
                end
            end
        end
    end

    % Filter the modes based on the keep_indices
    modes_filtered = modes(:, keep_indices);
end