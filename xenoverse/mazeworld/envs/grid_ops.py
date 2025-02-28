import numpy
from numpy import random
from attr import attrs, attrib
from xenoverse.mazeworld.envs.utils import conv2d_numpy

"""
Generating Topology For A Maze
"""

@attrs
class Rectangle(object):
    lb = attrib(type=list, default=None)
    rt = attrib(type=list, default=None)

    def resample(self, cells, max_size=10, min_size=2):
        m_x, m_y = cells.shape

        # Sample the rectangle width and length first
        w_x = random.randint(min_size, max_size + 1)
        w_y = random.randint(min_size, max_size + 1)

        kernel = numpy.ones((w_x, w_y), dtype="float32")

        # Use convolution to make sure no overlapping in rectangles
        overlap = conv2d_numpy(cells, kernel, stride=(1, 1))
        freerows, freecols = numpy.where(overlap < 0.5)

        if(freerows.shape[0] > 0):
            sel_idx = random.randint(0, freerows.shape[0])
        else:
            return False

        self.lb = [freerows[sel_idx], freecols[sel_idx]]
        self.rt = [self.lb[0] + w_x - 1, self.lb[1] + w_y - 1]
        return True

    def refresh_rectangle(self, cells):
        m_x, m_y = cells.shape
        b_x = max(0, self.lb[0] - 1)
        b_y = max(0, self.lb[1] - 1)
        e_x = min(m_x, self.rt[0] + 2)
        e_y = min(m_y, self.rt[1] + 2)
        cells[b_x:e_x, b_y:e_y] = 1

    def refresh_occupancy(self, cells):
        self.refresh_rectangle(cells)
        cells[self.lb[0]:(self.rt[0] + 1), self.lb[1]:(self.rt[1] + 1)] = 0

def genmaze_largeroom(n, room_number, room_size=(2,4)):
    cells_occ = numpy.zeros((n-2, n-2), dtype=numpy.int8)
    cells_wall = numpy.ones((n-2, n-2), dtype=numpy.int8)
    rects = []

    # TRY PUT 6 RECTANGLES
    max_try = 5
    for _ in range(room_number):
        rect = Rectangle()
        for _ in range(max_try):
            is_succ = rect.resample(cells_occ, min_size=room_size[0], max_size=room_size[1])
            if(is_succ):
                rect.refresh_rectangle(cells_occ)
                rects.append(rect)
                break
    for rect in rects:
        rect.refresh_occupancy(cells_wall)

    cell_occs = numpy.ones((n, n), dtype=numpy.int8)
    cell_walls = numpy.ones((n, n), dtype=numpy.int8)
    cell_occs[1:n-1, 1:n-1] = cells_occ
    cell_walls[1:n-1, 1:n-1] = cells_wall

    return cell_occs, cell_walls, rects

def genmaze_by_primwall(n, allow_loops=True, wall_density=0.30):
    # Dig big rooms in the region
    cell_occs, cell_walls, rects = genmaze_largeroom(n, random.randint(0, (n - 2) ** 2 // 16))

    # Dig the initial holes
    for i in range(1, n, 2):
        for j in range(1, n, 2):
            if(not cell_occs[i,j]):
                cell_walls[i,j] = 0

    #Initialize the logics for prim based maze generation
    wall_dict = dict()
    path_dict = dict()
    rev_path_dict = dict()
    path_idx = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if(cell_walls[i,j]): # we will keep the axial point
                wall_dict[i, j] = 0
            elif(not cell_occs[i,j]):
                path_dict[i,j] = path_idx
                rev_path_dict[path_idx] = [(i,j)]
                path_idx += 1
    for rect in rects:
        xb = rect.lb[0] + 1
        yb = rect.lb[1] + 1
        xe = rect.rt[0] + 2
        ye = rect.rt[1] + 2
        rev_path_dict[path_idx] = []
        for i in range(xb, xe):
            for j in range(yb, ye):
                path_dict[i,j] = path_idx
                rev_path_dict[path_idx].append((i, j))
        path_idx += 1

    #Prim the wall until all points are connected
    max_cell_walls = numpy.prod(cell_walls[1:-1, 1:-1].shape)
    while len(rev_path_dict) > 1 or (allow_loops and numpy.sum(cell_walls[1:-1, 1:-1]) > max_cell_walls * wall_density):
        wall_list = list(wall_dict.keys())
        random.shuffle(wall_list)
        for i, j in wall_list:
            new_path_id = -1
            connected_path_id = dict()
            abandon_path_id = dict()
            max_duplicate = 1

            for d_i, d_j in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if((d_i > 0  and d_i < n and d_j > 0 and d_j < n)
                        and cell_walls[d_i, d_j] < 1):
                    # calculate duplicate path id that might creat a loop
                    if path_dict[d_i, d_j] not in connected_path_id:
                        connected_path_id[path_dict[d_i, d_j]] = 1
                    else:
                        connected_path_id[path_dict[d_i, d_j]] += 1
                    if(connected_path_id[path_dict[d_i, d_j]] > max_duplicate):
                        max_duplicate = connected_path_id[path_dict[d_i, d_j]]

                    # decide the new path_id and find those to be deleted
                    if(path_dict[d_i, d_j] < new_path_id or new_path_id < 0):
                        if(new_path_id >= 0):
                            abandon_path_id[new_path_id] = (new_i, new_j)
                        new_path_id = path_dict[d_i, d_j]
                        new_i = d_i
                        new_j = d_j
                    elif(path_dict[d_i, d_j] != new_path_id): # need to be abandoned
                        abandon_path_id[path_dict[d_i, d_j]] = (d_i, d_j)
            if(len(abandon_path_id) >= 1 and max_duplicate < 2):
                break
            if(len(abandon_path_id) >= 1 and max_duplicate > 1 and allow_loops):
                break
            if(allow_loops and len(rev_path_dict) < 2 and random.random() < 0.2):
                break

        if(new_path_id < 0):
            continue

        # add the released wall
        rev_path_dict[new_path_id].append((i,j))
        path_dict[i,j] = new_path_id
        cell_walls[i,j] = 0
        del wall_dict[i,j]

        # merge the path
        for path_id in abandon_path_id:
            rev_path_dict[new_path_id].extend(rev_path_dict[path_id])
            for t_i_o, t_j_o in rev_path_dict[path_id]:
                path_dict[t_i_o,t_j_o] = new_path_id
            del rev_path_dict[path_id]
    return cell_walls

if __name__=="__main__":
    print(genmaze_by_primwall(15).astype("int8"))
