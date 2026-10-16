function finsound = generate_impact_from_materials(material1,material2,size1,size2,mass,stiffness)
%generate_impact_from_materials("wood","wood",1,2,1,1e7)
fs = 44100;
vel = 1;
max_comp = 3;
root = fileparts(fileparts(mfilename('fullpath')));
irFolder = fullfile(root,'data','resynth1');
    IR1 = audioread(string(irFolder)+"/"+material1+string(size1) +"/" +randi(20) +".wav");
    IR2 = audioread(string(irFolder)+"/"+material2+string(size2) +"/" +randi(20) +".wav");

    sound1 = generate_impact_sound(IR1,fs,mass,stiffness,vel,max_comp,1);
    sound2 = generate_impact_sound(IR2,fs,mass,stiffness,vel,max_comp,1);

    finsound = sound1 +sound2;

    sound(finsound,fs)

end