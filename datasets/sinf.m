% Generating the training and testing
% datasets for the trigonometric sin 
% function approximation.
t=[0:0.01:2*pi];     % These are the input values.
a=sin(t);               % These are the expected output values.


% Creating two new files called "sin.data"
% and "sin_test.data" that contain the values
% for the input and output values for the sin
% function f(x) = sin(x).
fileID1=fopen('sin.data','w');
fileID2=fopen('sin_test.data','w');

% Wrtiting the size of total number of sample data
% at the first line of the file and the total number
% of input signals at the second line of the file.
fprintf(fileID1,'%d\n',length(t));
fprintf(fileID1,'2\n',length(t));
fprintf(fileID2,'%d\n',length(t));
fprintf(fileID2,'1\n',length(t));


% Writting the values of the t,a arrays into
% the opened file streams.
for i=1:length(t)
    fprintf(fileID1,'%.3f %.3f\n',t(i),a(i));
    fprintf(fileID2,'%.3f\n',t(i));
end

% Closing file streams, and writing into
% the standard output stream a message 
% regarding the status of the data generation.
fclose(fileID1); fclose(fileID2);




epsilon=1e-12;      % Setting the converngence rate for the training process.
eta=0.09;           % Setting the learning rate for the training process.
momentum=0.004;      % Setting the momentum value fo the training process.
epochs=1000;         % Setting the total number of epochs.
alph=3.8;           % Setting the alpha coefficient value of the activation function.
bet=1.5;            % Setting the beta coefficient value of the activation function.



% Iterating over and incrementally increasing the number of neurons
% in the first hidden layer of the neural network.
for i=1:12

    % The following string will be formated and executed in the bash shell.
    command1="../neuralnet --train --curve-fitting --normalization=yes --in-file=sin.data --dump-dir=../sin --signals=1 --nlayers=2 --neurons-per-layer=[%d,1] --activation=htan --epsilon=%f --eta=%f --momentum=%f --epochs=%d --alpha=%f --beta=%f\n";
    
    % Formatting the string that will be executed in the command line.
    command1=sprintf(command1,i,epsilon,eta,momentum,epochs,alph,bet);

    % Executing the formated string in the unix shell
    % and ignoring the output data.
    [status,~]=unix(command1);
    
    % THe following string will be formated and executed in the bash shell.
    command2="../neuralnet --predict --curve-fitting --normalization=yes --in-file=sin_test.data --load-dir=../sin > sin_output.data";
    
    % Formatting the string that will be executed in the command line.
    comman2=sprintf(command2);

    % Executing the formated string in the unix shell
    % and ignoring the output data.
    [status,~]=unix(command2); 

    % Reading the predicted values from the
    % "quadratic_output.data" and plotting it
    % with the t,a so that we can have visualize
    % the goodness of the fit for the quadratic function.
    y=dlmread('sin_output.data');

    plot(t,a,'color','Cyan','0.2',t,y,'color','green','0.3');
    set(gca(),'color','black'); drawnow();
end

