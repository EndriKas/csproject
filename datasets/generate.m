t=[-1*pi:0.1:1*pi];
a=sin(t);
fileID=fopen('sin.data','w');

for i=1:length(t)
    fprintf(fileID,"%.3f  %.3f\n",t(i),a(i));
end
fclose(fileID);


% t=[0:0.1:20];
% a=t.^2;
% fileID=fopen('quadratic.data','w');

% for i=1:length(t)
%     fprintf(fileID,'%.3f %.3f\n',t(i),a(i))
% end

% fclose(fileID);

% t=[0:0.1:100];
% a=t.^3;
% fileID=fopen('cubic.data','w');
% for i=1:length(t)
%	fprintf(fileID,'%.3f %.3f\n',t(i),a(i))
% end


