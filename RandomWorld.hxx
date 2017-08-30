
#ifndef __RandomWorld_hxx
#define __RandomWorld_hxx

#include <World.hxx>
#include <Point2D.hxx>
#include <eigen3/Eigen/Sparse>
#include <vector>

typedef Eigen::SparseMatrix<float> SparseMatrixType; // declares a column-major sparse matrix type of float
typedef Eigen::Triplet<int> T;
typedef Eigen::Triplet<float> Tf;

namespace Examples 
{

class RandomWorldConfig;

class RandomWorld : public Engine::World
{
    //coordinates
    int _ij2val(Engine::Point2D<int>);
    Engine::Point2D<int> _val2ij(int);
    int _ji2val(Engine::Point2D<int>);
    Engine::Point2D<int> _val2ji(int);
    
    float _reward(Engine::Agent * ag);
    int chooseRandom(std::vector<double>);
    
    //initialization
    void createRasters();
    void createAgents();
    float L(int, int);
    float getActivationByAllAgents(Engine::Agent&);

public:	
    //rollout log
    std::vector<T> _pos_spr_coeff;
    std::vector<Tf> _rwd_spr_coeff;
    int hareCounter;
    int stagCounter;
    
    //uncontrolled dynamics
    std::vector<Tf> _q;
    std::vector<std::vector<float>>* _jointQ;
    std::vector<Tf>* _jointQ_sparse;
    void initQ();
    void initJointQ_twoAgents(std::vector<Tf> *);
    void initJointQ_twoAgents_old();
    float getQ(Engine::Point2D<int>, Engine::Point2D<int>);
    float getJointQ_sparse(int, int);
    
    //basis
    std::vector<Engine::Point2D<int>> basisCenters;
    void initBasis();
    std::vector<float> getPhiOfPos(Engine::Point2D<int>);
    std::vector<std::vector<float>> _phi;
    std::vector<double> theta;
    std::vector<float> phi_k;
    double logqpHat;
    double getQoverP();

    //action selection
    Engine::Point2D<int> getAction(Engine::Agent &);
    Engine::Point2D<int> getJointAction(Engine::Agent &);
    Engine::Point2D<int> getJointDistributedAction(Engine::Agent &);
    int computeJointAction(int);
    int nextJointAction;
    std::vector<int> nextJointAction_distr;
    
    //Pandora methods
    RandomWorld(Engine::Config * config, Engine::Scheduler * scheduler = 0);
    virtual ~RandomWorld();
    virtual void step();
	
};

} // namespace Examples 

#endif // __RandomWorld_hxx

