close all; clearvars

% Turn off the figrue visibility for all figures faster processing
set(0,'DefaultFigureVisible','off')

material = ["glass","ceramic","metal","stone","wood","plastic","cardboard"];

load_dir = "../../data/modes_synth/";
colorscheme = lines(length(material));
size_buckets = 3;
modes_all= [];

    close all

for ii =1:length(material)
for jjj = 1:length(size_buckets)
size_bucket= size_buckets(jjj);
load(load_dir+material(ii)+"_"+string(size_bucket)+"_modes.mat")

modes_all = [modes_all modes];
end

%generate domain for the contour plot 
freq_lims = [0,15000];
amp_lims = [-40,0];
RT60_lims = [0,0.4];
gridSize = 100;

[X,Y] = meshgrid(linspace(freq_lims(1),freq_lims(2),gridSize),linspace(amp_lims(1),amp_lims(2),gridSize));

% Get the kernel density estimate of the data
density = ksdensity(modes([1 2],:)',[X(:) Y(:)]);

density = reshape(density, gridSize, gridSize);
half_prob = max(density,[],"all")/2;

figure(1)
[~,l1{ii}] =contour(X, Y, density,[half_prob half_prob]);
hold on
l1{ii}.LineColor = colorscheme(ii,:);
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
title('Contour Plot: X-Y Plane');

% plot the 50% contour of the freq and RT60 data
[X,Y] = meshgrid(linspace(freq_lims(1),freq_lims(2),gridSize),linspace(RT60_lims(1),RT60_lims(2),gridSize));

% Get the kernel density estimate of the data
density = ksdensity(modes([1 3],:)',[X(:) Y(:)]);

density = reshape(density, gridSize, gridSize);
half_prob = max(density,[],"all")/2;

figure(2)
% plot only the contour at half_prob
[~,l2{ii}] = contour(X, Y, density,[half_prob half_prob]);
hold on
l2{ii}.LineColor = l1{ii}.LineColor;

xlabel('Frequency (Hz)');
ylabel('RT60(s)');
title('Contour Plot: X-Y Plane');

% plot a figure that shows the distribution of the third row of noise_modes with the first row of noise_modes
alpha = modes_noise(:,1:size(modes_noise,2)/2);
gamma = modes_noise(:,(size(modes_noise,2)/2 +1):end);
alph_freq = noise_modes(1,1:size(modes_noise,2)/2);

figure(5)
e1{ii}=plot(alph_freq,mean(alpha,1));
e1{ii}.Color = colorscheme(ii,:);
% Change the thickness of error bars
e1{ii}.LineWidth = 2;
xlabel('Frequency (Hz)');
ylabel('Alpha (dB)');
title('Alpha vs Frequency');

hold on

figure(6)
e1{ii}=plot(alph_freq,mean(gamma,1));
e1{ii}.Color = colorscheme(ii,:);
% Change the thickness of error bars
e1{ii}.LineWidth = 2;
hold on
xlabel('Frequency (Hz)');
ylabel('gamma (s)');
title('Gamma vs Frequency');

figure(1)
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
set(findall(gcf,'-property','LineWidth'),'LineWidth',2)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',10)
set(findall(gcf,'-property','MarkerFaceColor'),'MarkerFaceColor','auto')
set(findall(gcf,'-property','MarkerEdgeColor'),'MarkerEdgeColor','auto')
set(findall(gcf,'-property','Color'),'Color','w')
set(findall(gcf,'-property','Box'),'Box','on')

set(findall(gcf,'-property','Color'),'Color','w')
set(findall(gcf,'-property','Box'),'Box','on')
set(findall(gcf,'-property','XColor'),'XColor','k')
set(findall(gcf,'-property','YColor'),'YColor','k')
set(findall(gcf,'-property','ZColor'),'ZColor','k')
set(findall(gcf,'-property','GridColor'),'GridColor','k')
set(findall(gcf,'-property','MinorGridColor'),'MinorGridColor','k')
set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':')
set(findall(gcf,'-property','GridLineStyle'),'GridLineStyle',':')
set(findall(gcf,'-property','MinorGridAlpha'),'MinorGridAlpha',0.5)
set(findall(gcf,'-property','GridAlpha'),'GridAlpha',0.5)
set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':')

figure(2)
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
set(findall(gcf,'-property','LineWidth'),'LineWidth',2)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',10)
set(findall(gcf,'-property','MarkerFaceColor'),'MarkerFaceColor','auto')
set(findall(gcf,'-property','MarkerEdgeColor'),'MarkerEdgeColor','auto')
set(findall(gcf,'-property','Color'),'Color','w')
set(findall(gcf,'-property','Box'),'Box','on')

set(findall(gcf,'-property','Color'),'Color','w')
set(findall(gcf,'-property','Box'),'Box','on')
set(findall(gcf,'-property','XColor'),'XColor','k')
set(findall(gcf,'-property','YColor'),'YColor','k')
set(findall(gcf,'-property','ZColor'),'ZColor','k')
set(findall(gcf,'-property','GridColor'),'GridColor','k')
set(findall(gcf,'-property','MinorGridColor'),'MinorGridColor','k')
set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':')
set(findall(gcf,'-property','GridLineStyle'),'GridLineStyle',':')
set(findall(gcf,'-property','MinorGridAlpha'),'MinorGridAlpha',0.5)
set(findall(gcf,'-property','GridAlpha'),'GridAlpha',0.5)
set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':')

figure(5)
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
set(findall(gcf,'-property','LineWidth'),'LineWidth',2)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',10)
set(findall(gcf,'-property','MarkerFaceColor'),'MarkerFaceColor','auto')
set(findall(gcf,'-property','MarkerEdgeColor'),'MarkerEdgeColor','auto')
set(findall(gcf,'-property','Box'),'Box','on')

set(findall(gcf,'-property','Box'),'Box','on')
set(findall(gcf,'-property','XColor'),'XColor','k')
set(findall(gcf,'-property','YColor'),'YColor','k')
set(findall(gcf,'-property','ZColor'),'ZColor','k')
set(findall(gcf,'-property','GridColor'),'GridColor','k')
set(findall(gcf,'-property','MinorGridColor'),'MinorGridColor','k')
set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':')
set(findall(gcf,'-property','GridLineStyle'),'GridLineStyle',':')
set(findall(gcf,'-property','MinorGridAlpha'),'MinorGridAlpha',0.5)
set(findall(gcf,'-property','GridAlpha'),'GridAlpha',0.5)
set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':')

figure(6)
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','FontName'),'FontName','Helvetica')
set(findall(gcf,'-property','LineWidth'),'LineWidth',2)
set(findall(gcf,'-property','MarkerSize'),'MarkerSize',10)
set(findall(gcf,'-property','MarkerFaceColor'),'MarkerFaceColor','auto')
set(findall(gcf,'-property','MarkerEdgeColor'),'MarkerEdgeColor','auto')
set(findall(gcf,'-property','Box'),'Box','on')

set(findall(gcf,'-property','Box'),'Box','on')
set(findall(gcf,'-property','XColor'),'XColor','k')
set(findall(gcf,'-property','YColor'),'YColor','k')
set(findall(gcf,'-property','ZColor'),'ZColor','k')
set(findall(gcf,'-property','GridColor'),'GridColor','k')
set(findall(gcf,'-property','MinorGridColor'),'MinorGridColor','k')
set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':')
set(findall(gcf,'-property','GridLineStyle'),'GridLineStyle',':')
set(findall(gcf,'-property','MinorGridAlpha'),'MinorGridAlpha',0.5)
set(findall(gcf,'-property','GridAlpha'),'GridAlpha',0.5)
set(findall(gcf,'-property','MinorGridLineStyle'),'MinorGridLineStyle',':')

save_dir = load_dir+"figs4/averaged/";

 if ~exist(save_dir, 'dir')
     mkdir(save_dir);  % create the directory if it doesn't exist
 end

saveas(5,"../../Figures/all_figs/SupFigure6_1.svg")
saveas(6,"../../Figures/all_figs/SupFigure6_2.svg")

end
