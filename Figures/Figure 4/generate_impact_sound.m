function imp_s = generate_impact_sound(ir,audio_sr,mass,stiff,vel,max_comp,nonlin_flag)

         t_imp = 0:1/audio_sr:pi*(mass/stiff)^0.5;
         t_imp = linspace(0,pi*(mass/stiff)^0.5,pi*(mass/stiff)^0.5*audio_sr+1);
        impulse  = vel*sin((stiff/mass)^0.5 *t_imp);
        if nonlin_flag
         impulse = max_comp*(tanh(impulse/max_comp));
        end
         
         imp_s = conv(ir,impulse);
         
         imp_s = 0.9*(imp_s - min(imp_s)) / (max(imp_s)-min(imp_s));
         imp_s = imp_s- mean(imp_s);

end