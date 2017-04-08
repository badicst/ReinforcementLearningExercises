%% 
% this script finds the optimal strategy for the match problem with a
% epsilon-soft policy Monte Carlo method (see: soft_policy_bj_Script.m) for
% the blackjak example.
% 
%%  Initialization
close all;
clc; 
n_experiments=1e4;
%rand('seed',0); % randn('seed',0); 
nStates       = 16;
eps=0.005;
nActions      = 2; % 1=>pick up one match; 2=>pick up two 
Q             = zeros(nStates,nActions);  % the initial action-value function
%pol_pi        = ones(1,nStates);       %original delete
pol_pi        = 0.5*ones(nStates,nActions)
firstSARewSum = zeros(nStates,nActions);% for accumulating returns for each state-action pair 
firstSARewCnt = zeros(nStates,nActions);%counts the number of times, a state-action pair occurs up to current episode 

%% starting new episode
tic
for hi=1:n_experiments
  display(hi);
  n_matches     = nStates-1;
  stateseen = []; pol_taken = []; 

      
  while (n_matches > 0)
    state=n_matches+1;
    %display(state);
    stateseen(end+1)=state;
    if (n_matches > 1)
      pol_to_take  = sample_discrete( pol_pi(state,:), 1, 1 );  % value in {1,2}
    else
      pol_to_take = 1;
      rew=1;
    end;
    pol_taken(end+1) = pol_to_take;
    
    if (n_matches == pol_to_take)
      rew=1;
    else
      rew=0;
    end;
    n_matches = n_matches - pol_to_take;
    opponent_action=unidrnd(2);                                % random opponent
    %opponent_action=max(mod(n_matches-player_action,3),1);       %  smart opponent
    n_matches=max(n_matches-opponent_action,0);
    
  end;  %end of current episode
  %%
  % Now the state-weight array pairs, determined by elements of the state_seen 
  % array and the associate actions determined by the pol_pi array, all
  % have the same return ( which is equal, in this case, to 'rew'). These
  % returns.  
  %%
  %% Accumulate Rturns
  %1. Accumulate returns for the respective state-weight array pairs visited in this
  %episode in the firstSARewSum matrix. 2. Compute action-value function
  %Q(s,a) as the average of returns, and 3. compute epsilon-soft-greedy policy:
  %
  for si=1:length(stateseen),
    %display(si);
    if( stateseen(si) > 0 ) % we don't count "initial" and terminal states
      staInd = stateseen(si); 
      actInd = pol_taken(si); 
      firstSARewCnt(staInd,actInd) = firstSARewCnt(staInd,actInd)+1; 
      firstSARewSum(staInd,actInd) = firstSARewSum(staInd,actInd)+rew; 
      Q(staInd,actInd)             = firstSARewSum(staInd,actInd)/firstSARewCnt(staInd,actInd); % <-take the average 
      if (max( Q(staInd,:) ) ~= min( Q(staInd,:) ) )
        [dum,greedyChoice]           = max( Q(staInd,:) );
        notGreedyChoice              = 3-greedyChoice; % linear function that maps 1=>2 and 2=>1 
        % perform the eps-soft on-policy MC update: 
        pol_pi(staInd,greedyChoice)    = 1 - eps + eps/nActions; 
        pol_pi(staInd,notGreedyChoice) = eps/nActions;
      end;
    end;
  end;  %end of for loop
  
  %% Note:
  % The penultimate state is not a real state, because the player and the
  % opponent together must pick at least two matches, so this state cannot
  % be considered as a real state where the system can transition to.
  % Therefore there is no a clear policy for that state.
      
end;
toc
display(pol_pi);

%%%%%%
v=zeros(1,nStates);
v2=zeros(1,nStates);
for i=1:length(v)
  [val,greedyChoice]           = max( Q(i,:) );
  action=greedyChoice;
  v(i)=Q(i,action);
  v2(i)=val;
end,
display(v);
display(v2);