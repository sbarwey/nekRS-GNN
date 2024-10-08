#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "trajGen.hpp"
#include "gnn.hpp"
#include <cstdlib>
#include <filesystem>


void deleteDirectoryContents(const std::filesystem::path& dir)
{
    for (const auto& entry : std::filesystem::directory_iterator(dir))
        std::filesystem::remove_all(entry.path());
}


trajGen_t::trajGen_t(nrs_t *nrs_, int dt_factor_, dfloat time_init_)
{
    nrs = nrs_; // set nekrs object
    mesh = nrs->meshV; // set mesh object
    dt_factor = dt_factor_; 
    time_init = time_init_; 

    // set MPI rank and size 
    MPI_Comm &comm = platform->comm.mpiComm;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // allocate memory 
    dlong N = mesh->Nelements * mesh->Np; // total number of nodes

    if (verbose) printf("\n[RANK %d] -- Finished instantiating trajGen_t object\n", rank);
    if (verbose) printf("[RANK %d] -- The number of elements is %d \n", rank, mesh->Nelements);
}

trajGen_t::~trajGen_t()
{
    if (verbose) printf("[RANK %d] -- trajGen_t destructor\n", rank);
}

void trajGen_t::trajGenSetup()
{
    if (verbose) printf("[RANK %d] -- in trajGenSetup() \n", rank);
    if (write)
    {
        std::string irank = "_rank_" + std::to_string(rank);
        std::string nranks = "_size_" + std::to_string(size);
        std::filesystem::path currentPath = std::filesystem::current_path();
        currentPath /= "traj";
        writePath = currentPath.string();
        int poly_order = mesh->Nq - 1;
        writePath = writePath + "_poly_" + std::to_string(poly_order) 
                    + "/tinit_" + std::to_string(time_init)
                    + "_dtfactor_" + std::to_string(dt_factor)
                    + "/data" + irank + nranks;
        if (!std::filesystem::exists(writePath))
        {
            std::filesystem::create_directories(writePath);
        }
        else
        {
            deleteDirectoryContents(writePath);
        }
        MPI_Comm &comm = platform->comm.mpiComm;
        MPI_Barrier(comm);
    }
}

void trajGen_t::trajGenWrite(dfloat time, int tstep)
{
    if (write)
    {
        if (verbose) printf("[RANK %d] -- in trajGenWrite() \n", rank);
        
        // ~~~~ Write the data
        if ((tstep%dt_factor)==0)
        {
            nek::ocopyToNek(time, tstep);
            // print stuff
            if (platform->comm.mpiRank == 0) {
                if (verbose) printf("[TRAJ WRITE] -- In tstep %d, at physical time %g \n", tstep, time);
            }
            // write data
            writeToFileBinary(writePath + "/u_step_" + std::to_string(tstep) + ".bin",
                    nrs->U, nrs->fieldOffset, 3);
            writeToFileBinary(writePath + "/p_step_" + std::to_string(tstep) + ".bin",
                    nrs->P, nrs->fieldOffset, 1);
        }
    }
}
