clearvars; close all
load_dir = "obo5_more_sin/resynth_exp7new_v1t_try2/";
Fs =44100;
conds = 1:7;
materials = ["ceramic","glass","metal","stone","wood"];
size_buckets = 1:3;
masses = [0.1 10^-0.75 10.^(linspace(-0.5,1,4))];
mass2_all = masses(1)*ones([1 length(masses)-1]);
mass1_all = masses(2:end);
mass_conds =length(mass1_all);

amps =[0.5; 0.8; 0.2; 0.99; 0.5; 0.97; 0.3];
damps_cond = [0,0.25,0.5,0.75,0.9];

for iter = 1:200
    count =0;
    
    
for iii = 1:length(materials)
    material1 = materials(iii);
    material2 = materials(randi(length(materials)));
    
for jjj = conds
    for jkk = 1:length(size_buckets)
        size_bucket = size_buckets(jkk);
    for kkk = 1:length(damps_cond)
 

                modes1 = load(load_dir+num2str(jjj)+"/"+num2str(damps_cond(1))+"/"+num2str(jkk)+"/"+material1+"/all_modes.mat");
                modes2= load(load_dir+num2str(jjj)+"/"+num2str(damps_cond(kkk))+"/"+num2str(jkk)+"/"+material2+"/all_modes.mat");
                 count = count+1;



            RT60_1 = squeeze(modes1.all_modes(randi(50),:,3));
            RT60_2 = squeeze(modes2.all_modes(randi(50),:,3));


         


            savedir = load_dir + "expt_stim_exp7new_v1t_try2/"+num2str(iter)+"/";

            if ~exist(savedir, 'dir')
                     mkdir(savedir);  % create the directory if it doesn't exist
            end
            save(savedir+num2str(count)+".mat",'RT60_1','RT60_2')
            
            stim_key{count,1} = jjj;
            stim_key{count,2} = material1;
            stim_key{count,3} = kkk;
            stim_key{count,4} = size_bucket;
            stim_key{count,5} = material2;

        
    end
    end
end

end
% save stim_key as a csv in savedir
writecell(stim_key,savedir+"stim_key.csv");

stim_save=jsonencode([stim_key{:,2}]);
% Save the JSON string to a file
fid = fopen(savedir+"stim_key.json", 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, stim_save, 'char');
fclose(fid);

end

