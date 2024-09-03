#if !defined(nekrs_trajGen_hpp_)
#define nekrs_trajGen_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include <filesystem>

void deleteDirectoryContents(const std::filesystem::path& dir);


class trajGen_t 
{
public:
    trajGen_t(nrs_t *nrs, int dt_factor_, dfloat time_init_);
    ~trajGen_t(); 

    // member functions 
    void trajGenSetup();
    void trajGenWrite(dfloat time, int tstep);

    // where trajectory output files are written
    std::string writePath;
    
    // trajectory initial time 
    dfloat time_init; 

    // dt write factor (timestep interval as multiple of simulation timestep) 
    int dt_factor; 

private:
    // nekrs objects 
    nrs_t *nrs;
    mesh_t *mesh;
    ogs_t *ogs;

    // MPI stuff 
    int rank;
    int size;

    // for prints 
    bool verbose = true; 

    // for writing 
    bool write = true;
};

#endif
