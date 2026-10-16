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

% Create a single figure with 4x3 subplot layout with square subplots
figure('Units', 'inches', 'Position', [1 1 11 11]); % Made square by adjusting height to 11

for jjj = 1:length(size_buckets)
    size_bucket = size_buckets(jjj);
    
    % RT60 vs Frequency
    subplot(4, 3, 3*(jjj-1) + 1); hold on;
    for ii = 1:length(material)
        load(load_dir + material(ii) + "_" + string(size_bucket) + "_modes.mat");
        
        % --- Filter out potential replica modes ---
        modes_filtered = filter_modes(modes,material(ii)); 

        amp_norm = (modes_filtered(2,:) - min(modes_filtered(2,:))) / (max(modes_filtered(2,:)) - min(modes_filtered(2,:)));
        amp_norm(isnan(amp_norm)) = 0;
        transparency = tanh(0.4 + 0.6 * amp_norm);
        transparency(transparency > 1) = 1;

        s = scatter(modes_filtered(1,:), modes_filtered(3,:), abs(modes_filtered(2,:) + 50).^1, colorscheme(ii,:), 'filled');
        s.AlphaData = transparency;
        s.MarkerEdgeAlpha = 'flat';
        s.MarkerFaceAlpha = 'flat';
        
        l1(ii) = scatter(NaN, NaN, 50, colorscheme(ii,:), "filled"); 
        set(gca,'xlim',freq_lims);
        set(gca,'ylim',RT60_lims);
    end
    if jjj == 1
    end
    hold off;
    set(gca, 'PlotBoxAspectRatio', [1 1 1]);
    
    % Amplitude vs Frequency
    subplot(4, 3, 3*(jjj-1) + 2); hold on;
    for ii = 1:length(material)
        load(load_dir + material(ii) + "_" + string(size_bucket) + "_modes.mat");
        
        % --- Filter out potential replica modes ---
        modes_filtered = filter_modes(modes,material(ii)); 
    
        rt60_norm = (modes_filtered(3,:) - min(modes_filtered(3,:))) / (0.2 - min(modes_filtered(3,:)));
        rt60_norm(isnan(rt60_norm)) = 0;
        marker_size = 15 + rt60_norm * 29;
        transparency = tanh(0.4 + rt60_norm * 0.6); 
    
        s = scatter(modes_filtered(1,:), modes_filtered(2,:), marker_size, colorscheme(ii,:), 'filled');
        s.AlphaData = transparency;
        s.MarkerEdgeAlpha = 'flat';
        s.MarkerFaceAlpha = 'flat';
        
        l2(ii) = scatter(NaN, NaN, 50, colorscheme(ii,:), "filled"); 
        set(gca,'xlim',freq_lims);
        set(gca,'ylim',amp_lims);
    end
    hold off;
    set(gca, 'PlotBoxAspectRatio', [1 1 1]);
    
    % Amplitude vs RT60
    subplot(4, 3, 3*(jjj-1) + 3); hold on;
    for ii = 1:length(material)
        load(load_dir + material(ii) + "_" + string(size_bucket) + "_modes.mat");
    
        % --- Filter out potential replica modes ---
        modes_filtered = filter_modes(modes,material(ii));
    
        freq_norm = (modes_filtered(1,:) - min(modes_filtered(1,:))) / (max(modes_filtered(1,:)) - min(modes_filtered(1,:)));
        freq_norm(isnan(freq_norm)) = 0;
        marker_size = 25 + freq_norm * 19; 
        transparency = tanh(0.5 + freq_norm * 0.5); 
    
        s = scatter(modes_filtered(3,:), modes_filtered(2,:), marker_size, colorscheme(ii,:), 'filled');
        s.AlphaData = transparency;
        s.MarkerEdgeAlpha = 'flat';
        s.MarkerFaceAlpha = 'flat';
        
        l3(ii) = scatter(NaN, NaN, 50, colorscheme(ii,:), "filled"); 
        set(gca,'xlim',RT60_lims);
        set(gca,'ylim',amp_lims);
    end
    hold off;
    set(gca, 'PlotBoxAspectRatio', [1 1 1]);
    
    if jjj == length(size_buckets)
        % RT60 vs Frequency
        subplot(4, 3, 10); hold on;
        for ii = 1:length(material)
            load(load_dir + material(ii) + "_" + string(size_bucket) + "_modes.mat");
            
            % --- Filter out potential replica modes ---
            modes_filtered = filter_modes(modes,material(ii)); 

            amp_norm = (modes_filtered(2,:) - min(modes_filtered(2,:))) / (max(modes_filtered(2,:)) - min(modes_filtered(2,:)));
            amp_norm(isnan(amp_norm)) = 0;
            transparency = tanh(0.4 + 0.6 * amp_norm);
            transparency(transparency > 1) = 1;

            s = scatter(modes_filtered(1,:), modes_filtered(3,:), abs(modes_filtered(2,:) + 50).^1, colorscheme(ii,:), 'filled');
            s.AlphaData = transparency;
            s.MarkerEdgeAlpha = 'flat';
            s.MarkerFaceAlpha = 'flat';
            
            l1(ii) = scatter(NaN, NaN, 50, colorscheme(ii,:), "filled"); 
            set(gca,'xlim',freq_lims);
            set(gca,'ylim',RT60_lims);
        end
        hold off;
        set(gca, 'PlotBoxAspectRatio', [1 1 1]);

        % Amplitude vs Frequency
        subplot(4, 3, 11); hold on;
        for ii = 1:length(material)
            load(load_dir + material(ii) + "_" + string(size_bucket) + "_modes.mat");
            
            % --- Filter out potential replica modes ---
            modes_filtered = filter_modes(modes,material(ii)); 
    
            rt60_norm = (modes_filtered(3,:) - min(modes_filtered(3,:))) / (0.2 - min(modes_filtered(3,:)));
            rt60_norm(isnan(rt60_norm)) = 0;
            marker_size = 15 + rt60_norm * 29;
            transparency = tanh(0.4 + rt60_norm * 0.6); 
    
            s = scatter(modes_filtered(1,:), modes_filtered(2,:), marker_size, colorscheme(ii,:), 'filled');
            s.AlphaData = transparency;
            s.MarkerEdgeAlpha = 'flat';
            s.MarkerFaceAlpha = 'flat';
            
            l2(ii) = scatter(NaN, NaN, 50, colorscheme(ii,:), "filled"); 
            set(gca,'xlim',freq_lims);
            set(gca,'ylim',amp_lims);
        end
        hold off;
        set(gca, 'PlotBoxAspectRatio', [1 1 1]);

        % Amplitude vs RT60
        subplot(4, 3, 12); hold on;
        for ii = 1:length(material)
            load(load_dir + material(ii) + "_" + string(size_bucket) + "_modes.mat");
        
            % --- Filter out potential replica modes ---
            modes_filtered = filter_modes(modes,material(ii));
    
            freq_norm = (modes_filtered(1,:) - min(modes_filtered(1,:))) / (max(modes_filtered(1,:)) - min(modes_filtered(1,:)));
            freq_norm(isnan(freq_norm)) = 0;
            marker_size = 25 + freq_norm * 19; 
            transparency = tanh(0.5 + freq_norm * 0.5); 
    
            s = scatter(modes_filtered(3,:), modes_filtered(2,:), marker_size, colorscheme(ii,:), 'filled');
            s.AlphaData = transparency;
            s.MarkerEdgeAlpha = 'flat';
            s.MarkerFaceAlpha = 'flat';
            
            l3(ii) = scatter(NaN, NaN, 50, colorscheme(ii,:), "filled"); 
            set(gca,'xlim',RT60_lims);
            set(gca,'ylim',amp_lims);
        end
        hold off;
        set(gca, 'PlotBoxAspectRatio', [1 1 1]);
    end
end
save_figure(load_dir, size_bucket, "combined_plots.svg");

function format_plot()
    set(findall(gcf,'-property','FontSize'),'FontSize',8); % Reduced font size
    set(findall(gcf,'-property','FontName'),'FontName','Helvetica');
    set(findall(gcf,'-property','LineWidth'),'LineWidth',2);
    set(findall(gcf,'-property','Color'),'Color','w');
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
    save_dir = load_dir + "figs_combined/";
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    saveas(gcf, save_dir + "combined_plots3.svg");
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