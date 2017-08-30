RandomWorld:
        _ij2val, _ji2val: 2D position to cell number
        _val2ij & _val2ji: cell number to 2D position
        _reward: compute reward for agent
        chooseRandom: choose action depending on transition probabilities
        createRasters & createAgents: initialize Pandora rasters and initial agent positions

        _pos_spr_coeff: store sequence of positions
        _rwd_spr_coeff: store sequence of obtained rewards

        _q, _jointQ, & _jointQ_sparse: uncontrolled dynamics
        initQ, initJointQ_twoAgents: computes the uncontrolled dynamic matrix
        getQ, getJointQ_sparse: returns transition probability between a pair of cells

        basisCenters: vector of basis positions
        initBasis: basis initialization (change positions here)

        _phi, phi_k: basis activations in this rollout
        theta: vector of current theta values

        logqpHat: temporal value for the log(q/p)
        nextJointAction, nextJointAction_distr: temporal selected actions
        
        getAction & getJointDistributedAction: returns new 2D position for independent agents
        getJointAction: returns new 2D position for each agent (working in pairs)
        computeJointAction: computes new 2D position for each agent (working in pairs)

