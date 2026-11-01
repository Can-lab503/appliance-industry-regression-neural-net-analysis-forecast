% Excerpt from the competition paper: feedforward neural-network fitting.
% Inputs X_bx / T_bx come from non-public industry data and are not included in this repository.
% 神经网络输入输出拟合
% 定义变量如下
%   X_bx - input data.（冰箱输入值）
%   T_bx - target data.（冰箱输出值）
x = X_bx;
t = T_bx;
trainFcn = 'trainlm';  %选Levenberg-Marquardt 方法进行训练
% 创建隐藏层
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize,trainFcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
% 将数据划分为训练集，验证集，测试集
net.divideFcn = 'dividerand';  % 随机划分数据
net.divideMode = 'sample';  
net.divideParam.trainRatio = 70/100;%训练集占比百分之七十
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'mse';  % 均方误差
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
% 训练神经网络
[net,tr] = train(net,x,t);
% 测试神经网络
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
% 重复计算
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);
% 验证神经网络
view(net)
if (false)
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    gensim(net);
end
