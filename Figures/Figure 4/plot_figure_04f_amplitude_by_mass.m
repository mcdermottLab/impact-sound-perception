clearvars; close all;

load_dir= "./wood_metal_pellet/";
fileList = dir(load_dir +"*.wav");

% Initialize a matrix to store the maximum amplitudes
maxAmplitudes = [];
maxAmps = NaN([length(fileList) 40]);
masses = [1.2 5.9 8.7 12 16.2 28.1];

% Loop through each WAV file
for i = 1:length(fileList)
    close all
    y_fin = []; maxAmplitudes =[]; maxAmplitudesPerFile =[];
    % Load the WAV file
    [audio, fs] = audioread(load_dir+fileList(i).name);

        count = 0;
        y= audio;
        sq_y = smoothdata(y.^2,"gaussian",400);
        %Find indices where sq_y increases suddenly
        diff_y = diff(sq_y)*44100;
        [pks,locs]= findpeaks(diff_y,'MinPeakHeight',0.001,MinPeakDistance=30000);

        for kk = 1:length(locs)
            y_fin1 = y((locs(kk)-3000):(locs(kk)+15000));
            
            y_fin{kk} = y_fin1;
            count = count+1;
            maxAmplitudesPerFile(count)=max(abs(y_fin1));
        end
    
    maxAmplitudes{i} = maxAmplitudesPerFile;
    ave_Amp(i) = mean(maxAmplitudes{i});
    std_Amp(i) = std(maxAmplitudes{i});
    maxAmps(i,1:length(maxAmplitudesPerFile)) = 20*log10(maxAmplitudesPerFile/2e-6);
    
end

removeLastFour = @(s) s(1:end-4);

newCellArray = cellfun(removeLastFour, {fileList.name}, 'UniformOutput', false);

figure;
set(gcf, 'Units', 'inches', 'Position', [1 1 4 3]);
errorbar(masses, ave_Amp, std_Amp, 'o', 'LineWidth', 2, 'Color', 'k', 'MarkerFaceColor', 'k');
hold on
xlabel('Mass of pellet(g)');
ylabel('Maximum amplitude (dB)');
title('Maximum amplitude vs mass of pellet');
set(gca,'FontSize',12)

% Plot the best fit line with dashed black style
p = polyfit(masses, ave_Amp, 1);
x = linspace(0,30,100);
y = polyval(p,x);
plot(x, y, '--k', 'LineWidth', 2);

% Calculate the correlation coefficient
r = corrcoef(masses, ave_Amp);
fprintf('The correlation coefficient is %.2f\n', r(1,2));
saveas(gcf, "../all_figs/Figure_4f.svg")
