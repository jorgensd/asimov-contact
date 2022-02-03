import dolfinx.io
from IPython import embed
import dolfinx_contact
from dolfinx_contact.cpp import add_ghost_cells, update_ghosts
import dolfinx.fem
import dolfinx.mesh
import dolfinx_cuas
from mpi4py import MPI
import numpy as np
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)


def locator(x):
    return np.isclose(x[0], 0)


mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
facets = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim - 1, locator)
cells = dolfinx.mesh.compute_incident_entities(mesh, facets, mesh.topology.dim - 1, mesh.topology.dim)
cmap = mesh.topology.index_map(mesh.topology.dim)
cells = cells[cells < cmap.size_local]
mt = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim, cells, np.full(cells.size, mesh.comm.rank,
                                                                   dtype=np.int32))
with dolfinx.io.XDMFFile(mesh.comm, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(mt)
mesh.topology.create_connectivity(mesh.topology.dim, 0)
partitioner = add_ghost_cells(mesh, cells)
print(mesh.comm.rank, partitioner)
assert(partitioner.num_nodes == cmap.size_local)
contact_mesh = update_ghosts(mesh, partitioner)

# print("Rank", mesh.comm.rank, "Old num_ghost", mesh.topology.index_map(mesh.topology.dim).num_ghosts,
#       "new_num_ghosts", contact_mesh.topology.index_map(contact_mesh.topology.dim).num_ghosts)
