
#ifndef __RandomWorldConfig_hxx__
#define __RandomWorldConfig_hxx__

#include <Config.hxx>
#include <Point2D.hxx>
#include <vector>

namespace Examples
{

class RandomWorldConfig : public Engine::Config
{	

public:
	int _numAgents;
    int _maxIt;
    int _maxRolls;
    int _numRewards;
    std::vector<Engine::Point2D<int>> _rewardPositions;
	int _rewardAreaSize;
    int _numBasisX;
    int _numBasisY;
    float _basisSigma;
    float _lambda;
    float _gamma;
    float _learningRate;
    int _timeHorizon;
    
	RandomWorldConfig( const std::string & xmlFile );
	virtual ~RandomWorldConfig();

	void loadParams();

	friend class RandomWorld;
};

} // namespace Examples

#endif // __RandomWorldConfig_hxx__

