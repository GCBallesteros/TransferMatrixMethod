from copy import deepcopy
from collections import OrderedDict

from transfermatrixmethod import multilayer_response


class Structure:
    def __init__(self, t_layers, n_layers, n_dict, n_left, n_right):
        self.t_layers = t_layers
        self.n_layers = n_layers
        self.n_left = n_left
        self.n_right = n_right
        self.n_dict = deepcopy(n_dict)

        if len(self.t_layers) != len(self.n_layers):
            raise ValueError(
                f"The number of layers in the structure"
                f"({len(self.t_layers)}) is different from the number of indices of"
                f"refraction ({len(self.n_layers)})"
            )

        if not all(x in self.n_dict for x in self.n_layers):
            raise ValueError(
                f"The indices of refraction {set(n_layers) - set(n_dict)} are not in "
                "the index of refraction dict."
            )

    def get_n_layers(self):
        return self.n_layers

    def get_t_layers(self):
        return self.t_layers

    def get_n_dict(self):
        return self.n_dict

    def __add__(self, right_structure):
        right_n_dict = deepcopy(right_structure.get_n_dict())

        new_dict = deepcopy(self.get_n_dict())
        for index in right_n_dict:
            if index in new_dict:
                left_value = new_dict[index]
                right_value = right_n_dict[index]

                if left_value != right_value:
                    raise ValueError(
                        "Can't merge index of refraction dictionaries"
                        f"with incompatible values for index {index}"
                    )
            else:
                new_dict[index] = right_n_dict[index]

        compound_structure = Structure(
            deepcopy(self.get_t_layers()) + deepcopy(right_structure.get_t_layers()),
            deepcopy(self.get_n_layers()) + deepcopy(right_structure.get_n_layers()),
            n_dict=new_dict,
            n_left=self.n_left,
            n_right=self.n_right,
        )
        return compound_structure

    def __len__(self):
        return len(self.get_t_layers())

    def __iter__(self):
        yield from zip(self.get_t_layers(), self.get_n_layers())

    def reverse(self):
        out_structure = Structure(
            deepcopy(self.get_t_layers()[::-1]),
            deepcopy(self.get_n_layers()[::-1]),
            n_dict=deepcopy(self.get_n_dict()),
            n_left=self.n_right,
            n_right=self.n_left,
        )

        return out_structure


class Slab(Structure):
    def __init__(self, thickness, gap_index, n_dict, n_left, n_right):
        self.thickness = thickness
        self.gap_index = gap_index
        self.n_dict = n_dict
        self.n_left = n_left
        self.n_right = n_right

        if not all(x in self.n_dict for x in [n_left, n_right, gap_index]):
            raise ValueError(
                "Some of the indices of refraction  are not in the index of refraction dict"
            )

    def get_n_layers(self):
        return [self.gap_index]

    def get_t_layers(self):
        return [self.thickness]

    def reverse(self):
        out_structure = Slab(
            thickness=self.thickness,
            gap_index=self.gap_index,
            n_dict=deepcopy(self.get_n_dict()),
            n_left=self.n_right,
            n_right=self.n_left,
        )

        return out_structure


def symmetrize_structure(structure, t_gap=0, gap_material=("air", 1.0)):
    if structure.n_right != structure.n_left:
        raise ValueError(
            "Structures with n_right != n_left can't be meaningully symmetrized"
        )

    if t_gap > 0:
        gap = Structure(
            t_layers=[t_gap],
            n_layers=[gap_material[0]],
            n_dict={gap_material[0]: gap_material[1]},
            n_left=1,
            n_right=1,
        )

        new_structure = deepcopy(structure) + gap + structure.reverse()
    else:
        new_structure = deepcopy(structure) + structure.reverse()

    return new_structure


def simulate_structure(structures, freq, theta, pol="s", alter=None):
    """alter: callable
    A function that takes a list of structures and/or tuples of structures and names
    together with theta.
    """
    # Build a dictionary of the substructures so that it can be passed to the alter
    # function. We do so because this makes the user code in the alter function easier
    # to write.
    structures_dict = OrderedDict()
    structure_counter = 0
    for structure in structures:
        if isinstance(structure, tuple):
            # In this case we use the first element of the tuple to associate a name
            # with the structure to make it easier to access from within the alter
            # function if required.
            structures_dict[structure[0]] = deepcopy(structure[1])
        else:
            structures_dict[structure_counter] = deepcopy(structure)

        structure_counter += 1

    if not alter is None:
        structures_dict, theta = alter(structures_dict, theta)

    # Grab the first/leftmost element of the structure
    index_left = structures_dict[next(iter(structures_dict))].n_left

    first_pass = True
    for structure in structures_dict:
        if first_pass:
            final_structure = structures_dict[structure]
            first_pass = False
        else:
            final_structure = final_structure + structures_dict[structure]

    index_right = final_structure.n_right

    final_structure_n_layers, final_structure_t_layers, final_structure_n_dict = (
        final_structure.get_n_layers(),
        final_structure.get_t_layers(),
        final_structure.get_n_dict(),
    )

    t_layers = [0] + final_structure_t_layers + [0]
    n_layers_keys = [index_left] + final_structure_n_layers + [index_right]
    index_layers = [final_structure_n_dict[index] for index in n_layers_keys]

    R, T = multilayer_response(pol, t_layers, index_layers, freq, theta)

    return R, T
