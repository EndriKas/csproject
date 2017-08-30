fileID=fopen("sin_pattern.data","w");
t=[-0.9:0.01:pi/10];
a=sin(t);
for i=1:length(t)
    fprintf(fileID,"%.3f %.3f %.3f\n",-1.0,t(i),a(i));
end
