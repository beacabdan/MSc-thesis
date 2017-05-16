
#include <RandomWorldConfig.hxx>

namespace Examples
{

RandomWorldConfig::RandomWorldConfig( const std::string & xmlFile ) : Config(xmlFile), _numAgents(0)
{
}

RandomWorldConfig::~RandomWorldConfig()
{
}

void RandomWorldConfig::loadParams()
{
	_numAgents = getParamInt( "numAgents", "value");
    _timeHorizon = getParamInt( "numSteps", "value");
    _numRewards = getParamInt( "extrarewards", "num");
    for (int i = 1; i <= _numRewards; i++)
    {
        std::string x = "x" + std::to_string(i);
        std::string y = "y" + std::to_string(i);
        _rewardPositions.push_back(Engine::Point2D<int> (getParamInt( "extrarewards", x), getParamInt( "extrarewards", y)));
    }
	_rewardAreaSize = getParamInt( "extrarewards", "size");
    _numBasisX = getParamInt( "numBasis", "xrange");
    _numBasisY = getParamInt( "numBasis", "yrange");
    _basisSigma = getParamFloat( "numBasis", "sigma");
    _lambda = getParamFloat( "lambda", "value");
    _learningRate = getParamFloat( "learningRate", "value");
    _maxIt = getParamInt( "learningLoop", "iterations");
    _maxRolls = getParamInt( "learningLoop", "rollouts");
}
	
} // namespace Examples

