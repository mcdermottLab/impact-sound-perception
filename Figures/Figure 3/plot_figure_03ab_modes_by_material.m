close all; clearvars

% Turn off the figure visibility for all figures faster processing
set(0,'DefaultFigureVisible','off')

material = ["wood","glass","metal","cardboard","plastic","ceramic","stone"];
size_buckets = 3;
load_dir = "../../data/modes_synth/";

% Bootstrap parameters
n_bootstrap = 100;  % number of bootstrap samples
bootstrap_alpha = 0.3;  % transparency for contours
colorscheme = lines(length(material));

close all

% Define size labels at the start
size_labels = ["100cm"];

% Create separate figures for each size bucket
for jjj = 1:length(size_buckets)
    figure(jjj*2-1)  % RT60 plots
    figure(jjj*2)    % Amplitude plots
end

for jjj = 1:length(size_buckets)
    size_bucket = size_buckets(jjj);
    
    % Create figures for this size
    figure_rt60 = jjj*2-1;
    figure_amp = jjj*2;
    
    for ii = 1:length(material)
        load(load_dir+material(ii)+"_"+string(size_bucket)+"_modes.mat")

        num_objects = size(modes,2)/10;
        % Setup domains
        freq_d = linspace(0,12000,100);
        amp_d = linspace(-40,0,100);
        RT60_d = linspace(0,0.5,100);
        [X, Y] = meshgrid(freq_d, RT60_d);

        modes_obj = reshape(modes, size(modes,1), [], num_objects);
        
        % Bootstrap loop for RT60 vs Frequency
        figure(figure_rt60)
        for b = 1:n_bootstrap
            % Random sampling with replacement
            boot_idx = randi(num_objects, [1, randi(num_objects)]);
            boot_modes = modes_obj(:,:,boot_idx);
            boot_modes = reshape(boot_modes, size(modes,1), []);
            
            mean_13 = nanmean(boot_modes([1 3],:),2);
            cov_13 = nancov(boot_modes([1 3],:)');
            
            pos = [X(:) Y(:)];
            gaussian = mvnpdf(pos, mean_13', cov_13);
            gaussian = reshape(gaussian, size(X));
            half_prob = max(gaussian,[],'all')*0.5;
            
            [~,l1(ii)] = contour(X, Y, gaussian(:,:),[half_prob half_prob]);
            l1(ii).LineColor = colorscheme(ii,:);  % Use material index for color
            l1(ii).LineWidth = 0.25;
            hold on
        end
        xlabel('Frequency (Hz)');
        ylabel('RT60 (s)');
        title(['Size: ' size_labels(jjj)])
        
        % Bootstrap loop for Amplitude vs Frequency
        [X, Y] = meshgrid(freq_d, amp_d);
        figure(figure_amp)
        for b = 1:n_bootstrap
            boot_idx = randi(size(modes,2), [1, size(modes,2)]);
            boot_modes = modes(:,boot_idx);
            
            mean_12 = nanmean(boot_modes([1 2],:),2);
            cov_12 = nancov(boot_modes([1 2],:)');
            
            pos = [X(:) Y(:)];
            gaussian = mvnpdf(pos, mean_12', cov_12);
            gaussian = reshape(gaussian, size(X));
            half_prob = max(gaussian,[],'all')*0.5;
            
            [~,l2(ii)] = contour(X, Y, gaussian(:,:),[half_prob half_prob]);
            l2(ii).LineColor = colorscheme(ii,:);  % Use material index for color
            l2(ii).LineWidth = 0.25;
            hold on
        end
        xlabel('Frequency (Hz)');
        ylabel('Amplitude (dB)');
        title(['Size: ' size_labels(jjj)])
    end
    
    % Add legends and formatting for each figure
    figure(figure_rt60)
    legend(l1, material)
    set(findall(gcf,'-property','FontSize'),'FontSize',20)
    set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
    set(findall(gcf,'-property','MarkerSize'),'MarkerSize',10)
    set(findall(gcf,'-property','MarkerFaceColor'),'MarkerFaceColor','auto')
    set(findall(gcf,'-property','MarkerEdgeColor'),'MarkerEdgeColor','auto')
    set(gcf,'Color','w')
    set(gca,'Color','w')
    set(findall(gcf,'-property','Box'),'Box','on')

    % Make the axes and grid lines black
    set(gca,'XColor','k')
    set(gca,'YColor','k')
    set(gca,'GridColor','k')
    set(gca,'MinorGridColor','k')
    set(gca,'MinorGridLineStyle',':')
    set(gca,'GridLineStyle',':')
    set(gca,'MinorGridAlpha',0.5)
    set(gca,'GridAlpha',0.5)

    figure(figure_amp)
    legend(l2, material)
    set(findall(gcf,'-property','FontSize'),'FontSize',20)
    set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
    set(findall(gcf,'-property','MarkerSize'),'MarkerSize',10)
    set(findall(gcf,'-property','MarkerFaceColor'),'MarkerFaceColor','auto')
    set(findall(gcf,'-property','MarkerEdgeColor'),'MarkerEdgeColor','auto')
    set(gcf,'Color','w')
    set(gca,'Color','w')
    set(findall(gcf,'-property','Box'),'Box','on')

    % Make the axes and grid lines black
    set(gca,'XColor','k')
    set(gca,'YColor','k')
    set(gca,'GridColor','k')
    set(gca,'MinorGridColor','k')
    set(gca,'MinorGridLineStyle',':')
    set(gca,'GridLineStyle',':')
    set(gca,'MinorGridAlpha',0.5)
    set(gca,'GridAlpha',0.5)

    saveas(figure_rt60, "../all_figs/Figure_3a.svg")
    saveas(figure_amp, "../all_figs/Figure_3b.svg")
    
end
