function out_flag = is_valid_mode_set(modes,M,C)


f1 = (sum(sign(modes(:,3)))==-1);
f2 = (sum(diff(modes(:,1))<10)==0);
f3 = 1;
f4 = (sum(modes(:,1)<500)>1);
f5 = (sum(modes(:,1)>2*M(1))==0);
f6 = sum(modes(:,3)>0.19)==0;

out_flag = ~f1 && f2 && f3 && f4 && f5 &&f6;


end