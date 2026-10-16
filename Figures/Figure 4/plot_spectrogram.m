function plot_spectrogram(y,Fs)

y = y(:,1);
nwin = 256; % Reduced from 512
wind = hanning(nwin);
nlap = round(nwin * 0.75);  % Increased overlap
nfft = 2^10;  % Reduced from 2^11

[s,f,t]=spectrogram(y,wind,nlap,nfft,Fs,'yaxis');

set(gcf, 'Units', 'inches', 'Position', [2, 2, 4, 3])
imagesc(t,f,10*log10(abs(s)))
ax = gca;
ax.YDir = 'normal';

set(gca,'ylim', [20 14000],'Fontsize',12)%,'YLabel',"Frequency(Hz)",'xlabel','Time')
xlabel('Time(s)'); ylabel('Frequency(Hz)'); 
% Create custom blue-white colormap
n = 256;  % number of colors
blue = [0 0.4 0.8];  % RGB values for blue
white = [1 1 1];     % RGB values for white
cmap = zeros(n,3);
for i = 1:3
    cmap(:,i) = linspace(white(i), blue(i), n);
end
colormap(cmap);
clim([-20 10])
cb1 = colorbar;
ylabel(cb1,'Sound Intensity/Frequency (dB/Hz)','Fontsize',12)
end
