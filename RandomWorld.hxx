
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
    int _ij2val(Engine::Point2D<int>);
    Engine::Point2D<int> _val2ij(int);
    int _ji2val(Engine::Point2D<int>);
    int _reward(Engine::Point2D<int>);
    void createRasters();
    void createAgents();
    float L(int, int);
    int chooseRandom(std::vector<double>);
    float getActivationByAllAgents(Engine::Agent&);

public:	
    std::vector<T> _pos_spr_coeff;
    std::vector<T> _rwd_spr_coeff;
    std::vector<Tf> _q;
    std::vector<Engine::Point2D<int>> basisCenters;
    std::vector<std::vector<float>> _phi;
    std::vector<double> theta;
    std::vector<float> phi_k;
    double logqpHat;

    Engine::Point2D<int> getAction(Engine::Agent &); 
    RandomWorld(Engine::Config * config, Engine::Scheduler * scheduler = 0);
    virtual ~RandomWorld();
    virtual void step();
    void initQ();
    float getQ(Engine::Point2D<int>, Engine::Point2D<int>);
    void initBasis();
    std::vector<float> getPhiOfPos(Engine::Point2D<int>);
    double getQoverP();
	
};

} // namespace Examples 

#endif // __RandomWorld_hxx

