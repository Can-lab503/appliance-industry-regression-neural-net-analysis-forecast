% Excerpt from the competition paper: NARX nonlinear autoregressive network for time-series forecasting.
% Inputs X_kt / T_kt come from non-public industry data and are not included in this repository.
% 定义变量
%   X_kt - input time series.（空调输入值）
%   T_kt - feedback time series.（空调输入值）

X = tonndata(X_kt,true,false);
T = tonndata(T_kt,true,false);

trainFcn = 'trainlm';  % 运用Levenberg-Marquardt 方法进行训练

% 建立一个外部输入为1:2的非线性自回归网络
inputDelays = 1:2;
feedbackDelays = 1:2;
hiddenLayerSize = 10;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};
[x,xi,ai,t] = preparets(net,X,{},T);

% 划分数据为训练集、测试集与验证集
net.divideFcn = 'dividerand';  %随机划分数据
net.divideMode = 'time';  
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'mse';  % 均方误差

net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

% 训练神经网络
[net,tr] = train(net,x,t,xi,ai);

% 测试神经网络
y = net(x,xi,ai);
e = gsubtract(t,y);
performance = perform(net,t,y)

trainTargets = gmultiply(t,tr.trainMask);
valTargets = gmultiply(t,tr.valMask);
testTargets = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% 验证神经网络
view(net)
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
view(netc)
[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc)

numTimesteps = size(x,2);
knownOutputTimesteps = 1:(numTimesteps-5);
predictOutputTimesteps = (numTimesteps-4):numTimesteps;
X1 = X(:,knownOutputTimesteps);
T1 = T(:,knownOutputTimesteps);
[x1,xio,aio] = preparets(net,X1,{},T1);
[y1,xfo,afo] = net(x1,xio,aio);
x2 = X(1,predictOutputTimesteps);
[netc,xic,aic] = closeloop(net,xfo,afo);
[y2,xfc,afc] = netc(x2,xic,aic);
multiStepPerformance = perform(net,T(1,predictOutputTimesteps),y2)
nets = removedelay(net);
nets.name = [net.name ' - Predict One Step Ahead'];
view(nets)
[xs,xis,ais,ts] = preparets(nets,X,{},T);
ys = nets(xs,xis,ais);
stepAheadPerformance = perform(nets,ts,ys)
if (false)
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x,xi,ai);
end
if (false)
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    x1 = cell2mat(x(1,:));
    x2 = cell2mat(x(2,:));
    xi1 = cell2mat(xi(1,:));
    xi2 = cell2mat(xi(2,:));
    y = myNeuralNetworkFunction(x1,x2,xi1,xi2);
end
if (false)
    gensim(net);
end
