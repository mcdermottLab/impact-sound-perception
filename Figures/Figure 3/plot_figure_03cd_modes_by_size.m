close all; clearvars

% Turn off the figure visibility for all figures faster processing
set(0,'DefaultFigureVisible','off')

material = ["metal"];
size_buckets = 1:4;
load_dir = "../../data/modes_synth/";

% Bootstrap parameters
n_bootstrap = 100;  % number of bootstrap samples
bootstrap_alpha = 0.3;  % transparency for contours
colorscheme = parula(length(size_buckets));

close all

% Define size labels at the start
size_labels = ["10cm", "30cm", "100cm", "1m"];

for ii = 1:length(material)
    close all
    for jjj = 1:length(size_buckets)
        size_bucket = size_buckets(jjj);
        
        load(load_dir+material(ii)+"_"+string(size_bucket)+"_modes.mat")
        
        % Setup domains
        freq_d = linspace(0,10000,100);
        amp_d = linspace(-40,0,100);
        RT60_d = linspace(0,0.4,100);
        [X, Y] = meshgrid(freq_d, RT60_d);

        % Bootstrap loop for RT60 vs Frequency
        figure(1)
        for b = 1:n_bootstrap
            % Random sampling with replacement
            boot_idx = randi(size(modes,2), [1, size(modes,2)]);
            boot_modes = modes(:,boot_idx);
            
            mean_13 = nanmean(boot_modes([1 3],:),2);
            cov_13 = nancov(boot_modes([1 3],:)');
            
            pos = [X(:) Y(:)];
            gaussian = mvnpdf(pos, mean_13', cov_13);
            gaussian = reshape(gaussian, size(X));
            half_prob = max(gaussian,[],'all')*0.5;
            
            [~,l1(jjj)] = contour(X, Y, gaussian(:,:),[half_prob half_prob]);
            l1(jjj).LineColor = colorscheme(jjj,:);
            l1(jjj).LineWidth = 0.25;
            hold on
        end
        xlabel('Frequency (Hz)');
        ylabel('RT60 (s)');

        % Bootstrap loop for Amplitude vs Frequency
        [X, Y] = meshgrid(freq_d, amp_d);
        figure(2)
        for b = 1:n_bootstrap
            boot_idx = randi(size(modes,2), [1, size(modes,2)]);
            boot_modes = modes(:,boot_idx);
            
            mean_12 = nanmean(boot_modes([1 2],:),2);
            cov_12 = nancov(boot_modes([1 2],:)');
            
            pos = [X(:) Y(:)];
            gaussian = mvnpdf(pos, mean_12', cov_12);
            gaussian = reshape(gaussian, size(X));
            half_prob = max(gaussian,[],'all')*0.5;
            
            [~,l2(jjj)] = contour(X, Y, gaussian(:,:),[half_prob half_prob]);
            l2(jjj).LineColor = colorscheme(jjj,:);
            l2(jjj).LineWidth = 0.25;
            hold on
        end
        xlabel('Frequency (Hz)');
        ylabel('Amplitude (dB)');

        %generate domain for the contour plot 
        freq_d = linspace(0,16000,300);
        amp_d = linspace(-60,20,100);

        mean_12 = nanmean(noise_modes([1 2],:),2);
        cov_12 = nancov(noise_modes([1 2],:)');
        RT60_d = linspace(0,0.3,100);

        [X, Y] = meshgrid(freq_d, RT60_d);
        mean_13 = nanmean(noise_modes([1 3],:),2);
        cov_13 = nancov(noise_modes([1 3],:)'); 

        pos = [X(:) Y(:)];
        gaussian = mvnpdf(pos, mean_13', cov_13);
        gaussian = reshape(gaussian, size(X));
        half_prob = max(gaussian,[],'all')*0.5;

        % plot the 50% contour of the freq and amp data
        [X, Y] = meshgrid(freq_d, amp_d);

        pos = [X(:) Y(:)];
        gaussian = mvnpdf(pos, mean_12', cov_12);
        gaussian = reshape(gaussian, size(X));
        half_prob = max(gaussian,[],'all')*0.5;

        % plot a figure that shows the distribution of the third row of noise_modes with the first row of noise_modes
        alpha = modes_noise(:,1:size(modes_noise,2)/2);
        gamma = modes_noise(:,(size(modes_noise,2)/2 +1):end);
        alph_freq = noise_modes(1,1:size(modes_noise,2)/2);

    end

    figure(1)
    legend(l1,size_labels)
    set(findall(gcf,'-property','FontSize'),'FontSize',20)
    set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
    set(findall(gcf,'-property','MarkerSize'),'MarkerSize',10)
    set(findall(gcf,'-property','MarkerFaceColor'),'MarkerFaceColor','auto')
    set(findall(gcf,'-property','MarkerEdgeColor'),'MarkerEdgeColor','auto')
    set(gcf,'Color','w')
    set(gca,'Color','w')
    set(findall(gcf,'-property','Box'),'Box','on')
    set(gca,"xlim",[0 12000],"ylim",[0 0.5])

    % Make the axes and grid lines black
    set(gca,'XColor','k')
    set(gca,'YColor','k')
    set(gca,'GridColor','k')
    set(gca,'MinorGridColor','k')
    set(gca,'MinorGridLineStyle',':')
    set(gca,'GridLineStyle',':')
    set(gca,'MinorGridAlpha',0.5)
    set(gca,'GridAlpha',0.5)

    figure(2)
    legend(l2,size_labels)
    set(findall(gcf,'-property','FontSize'),'FontSize',20)
    set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
    set(findall(gcf,'-property','MarkerSize'),'MarkerSize',10)
    set(findall(gcf,'-property','MarkerFaceColor'),'MarkerFaceColor','auto')
    set(findall(gcf,'-property','MarkerEdgeColor'),'MarkerEdgeColor','auto')
    set(gcf,'Color','w')
    set(gca,'Color','w')
    set(findall(gcf,'-property','Box'),'Box','on')
    set(gca,"xlim",[0 12000])

    % Make the axes and grid lines black
    set(gca,'XColor','k')
    set(gca,'YColor','k')
    set(gca,'GridColor','k')
    set(gca,'MinorGridColor','k')
    set(gca,'MinorGridLineStyle',':')
    set(gca,'GridLineStyle',':')
    set(gca,'MinorGridAlpha',0.5)
    set(gca,'GridAlpha',0.5)

saveas(1, "../all_figs/Figure_3c.svg")
    saveas(2, "../all_figs/Figure_3d.svg")

end
