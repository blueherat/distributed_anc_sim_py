function results = NWD_D(params)
    %% 1. 解包参数
    time            = params.time;
    rirManager      = params.rirManager;
    mu              = params.mu;
    x               = params.referenceSignal; % 参考信号
    d               = params.desiredSignal;   % 期望信号
    block_size      = params.block_size;
    overlap         = params.overlap;
    hop             = block_size - overlap;

    % 从rirManager获取参数
    keyPriSpks = keys(rirManager.PrimarySpeakers);
    keySecSpks = keys(rirManager.SecondarySpeakers);
    keyErrMics = keys(rirManager.ErrorMicrophones);

    numPriSpks      = numEntries(rirManager.PrimarySpeakers);
    numSecSpks      = numEntries(rirManager.SecondarySpeakers);
    numErrMics      = numEntries(rirManager.ErrorMicrophones);
    
    nSamples = length(time);

    %% 2. 初始化
    max_Ls_hat = 0;
    for i = keySecSpks'
        Ls_hat = length(rirManager.getSecondaryRIR(i, keyErrMics(1)));
        if Ls_hat > max_Ls_hat
            max_Ls_hat = Ls_hat;
        end
    end

    e = d; % 误差信号初始化

    x_buffer = zeros(block_size, numPriSpks);

    y_buffer = zeros(block_size, numSecSpks);

    e_buffer = zeros(block_size, numErrMics);
    %% 3. 主循环
    disp('开始NWD_D仿真...');
    for n = 1:hop:nSamples - block_size
        x_buffer = [x_buffer(hop+1:end, :); x(n:n+hop-1, :)];
        e_buffer = [e_buffer(hop+1:end, :); e(n:n+hop-1, :)];
    end
        
end