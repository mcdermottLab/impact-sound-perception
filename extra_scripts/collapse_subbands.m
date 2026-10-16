function signal = collapse_subbands(subbands, filts)

signal_length = length(subbands);
filt_length = size(filts,1);
if rem(signal_length,2)==0 %even length - 
    fft_filts = [filts' fliplr(filts(2:filt_length-1,:)')]'; %generate negative frequencies in right place; filters are column vectors
else %odd length
    fft_filts = [filts' fliplr(filts(2:filt_length,:)')]';
end
fft_subbands = fft_filts.*(fft(subbands));
subbands = ifft(fft_subbands);
signal = sum(subbands,2);
