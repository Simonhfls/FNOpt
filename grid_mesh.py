import torch
class GridMesh:
    def __init__(
        self,
        height: int,
        width: int,
        *,
        side_length_h: float = 1.0,
        side_length_w: float = 1.0,
        mass: float = 1.0,
    ):
        self.side_length_h = side_length_h
        self.side_length_w = side_length_w
        self.width = width
        self.height = height

        xx, yy, uu, vv = self.meshgrid()
        # self.pos_grid = torch.cat([xx.unsqueeze(0), yy.unsqueeze(0), torch.zeros(1, self.height, self.width)]).permute(1, 2, 0)
        self.pos_grid = torch.cat([xx.unsqueeze(0), yy.unsqueeze(0), torch.zeros(1, self.width, self.height)]).permute(1, 2, 0)
        self.uv_grid = torch.cat([uu.unsqueeze(0), vv.unsqueeze(0)]).permute(1, 2, 0)

        self.pos = self.pos_grid.flatten(0, 1)  # (num_vertices, 3)
        self.tri = self.generate_triangles()  # (num_triangles, 3)
        self.quad = self.generate_quads()  # (num_quads, 4)
        self.uv = self.uv_grid.flatten(0, 1)  # (num_vertices, 2)
        # self.faces_uv = ...
        self.mass = self.generate_mass(mass).flatten()  # (num_vertices, )

        self.num_vertices = self.pos.shape[0]
        self.num_triangles = self.tri.shape[0]

        self.area = 1 / 2 * (self.side_length_h * self.side_length_w) * torch.ones(self.num_triangles) # (num_triangles, )

        identity = torch.eye(2).expand(self.num_triangles, -1, -1)  # (num_triangles, 2, 2)
        self.metric_tensor = (self.side_length_h * self.side_length_w) * identity
        self.metric_tensor_inv = 1 / (self.side_length_h * self.side_length_w) * identity


    def meshgrid(self):
        # x_range = self.side_length * torch.linspace(1, 0, self.width)
        # y_range = self.side_length * torch.linspace(1, 0, self.height)
        x_range = self.side_length_w * torch.linspace(1, 0, self.width)
        y_range = self.side_length_h * torch.linspace(1, 0, self.height)
        xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')

        u_range = torch.linspace(1, 0, self.width)
        v_range = torch.linspace(1, 0, self.height)
        uu, vv = torch.meshgrid(u_range, v_range, indexing='ij')

        return xx, yy, uu, vv


    def generate_triangles(self):
        tri = []
        for i in range(self.width - 1):  # vertical
            for j in range(self.height - 1):  # horizontal
                bottom_left = i * self.height + j
                bottom_right = i * self.height + (j + 1)
                top_left = (i + 1) * self.height + j
                top_right = (i + 1) * self.height + (j + 1)

                # Split each quad into two triangles
                tri.append([bottom_left, bottom_right, top_left])
                tri.append([top_right, top_left, bottom_right])

        return torch.tensor(tri, dtype=torch.int32)


    def generate_quads(self):
        quad = []
        for i in range(self.width - 1):
            for j in range(self.height - 1):
                bottom_left = i * self.height + j
                bottom_right = i * self.height + (j + 1)
                top_left = (i + 1) * self.height + j
                top_right = (i + 1) * self.height + (j + 1)
                # Split the square element into two triangles
                quad.append([bottom_left, top_left, top_right, bottom_right])

        return torch.tensor(quad)


    def generate_mass(self, m):
        mass = m * torch.ones(self.height, self.width)
        mass[0, :] = mass[-1, :] = mass[:, 0] = mass[:, -1] = m / 2
        mass[0, 0] = mass[0, -1] = mass[-1, 0] = mass[-1, -1] = m / 4

        return mass


    def random_state(self, *, rotation, translation):
        """
        Args:
            rotation (torch.Tensor): Rotation matrices. Shape: (batch_size, 3, 3)
            translation (torch.Tensor): Translation vectors. Shape: (batch_size, 3)

        Returns:
            state (torch.Tensor): Deformed state. Shape: (batch_size, num_vertices, 3)
        """
        batch_size = rotation.shape[0]
        assert batch_size == translation.shape[0]

        pos = self.pos.unsqueeze(0).expand(batch_size, -1, -1)
        state = torch.bmm(pos, rotation.transpose(1, 2)) + translation.unsqueeze(1)
        return state


    def get_element_quantity(self):
        return self.area, self.metric_tensor_inv, self.tri

    def get_edge_length(self):
        return (self.pos[self.v1] - self.pos[self.v2]).norm(dim=-1)

    def get_adjacent_quantity(self):
        sharing_edge = self.pos[self.v2] - self.pos[self.v1]
        edge_length = sharing_edge.norm(dim=-1)
        # sharing_edge = sharing_edge / torch.norm(sharing_edge, dim=-1, keepdim=True)
        area_edge = 1 / 3 * (self.area[self.edge_tri[:, 0]] + self.area[self.edge_tri[:, 1]])

        return edge_length, area_edge

    def diheral_angle(self, pos):
        x1, x2, x3, x4 = pos[..., self.v1, :], pos[..., self.v2, :], pos[..., self.v3, :], pos[..., self.v4, :]
        sharing_edge = x2 - x1
        sharing_edge = sharing_edge / torch.norm(sharing_edge, dim=-1, keepdim=True)

        n1 = torch.cross(x2 - x1, x3 - x1, dim=-1)
        n1 = n1 / torch.norm(n1, dim=-1, keepdim=True)
        n2 = torch.cross(x2 - x1, x4 - x1, dim=-1)
        n2 = n2 / torch.norm(n2, dim=-1, keepdim=True)

        cross_product = torch.cross(n1, n2, dim=-1)
        dot_product = (n1 * n2).sum(dim=-1)

        return torch.pi - torch.atan2((sharing_edge * cross_product).sum(dim=-1), dot_product)  # DDG K. Crane

    def save_obj(self, filename, pos=None):
        if pos is None:
            pos = self.pos

        with open(filename, 'w') as file:
            for v in pos:
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")

            if self.uv is not None:
                for vt in self.uv:
                    file.write(f"vt {vt[0]} {vt[1]}\n")
                for f in self.tri:
                    file.write(f"f {f[0] + 1}/{f[0] + 1} {f[2] + 1}/{f[2] + 1} {f[1] + 1}/{f[1] + 1}\n")
            else:
                for f in self.tri:
                    file.write(f"f {f[0] + 1} {f[2] + 1} {f[1] + 1}\n")

    def save_quad_obj(self, filename, pos=None):
        if pos is None:
            pos = self.pos

        with open(filename, 'w') as file:
            for v in pos:
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")

            if self.uv is not None:
                for vt in self.uv:
                    file.write(f"vt {vt[0]} {vt[1]}\n")
                for f in self.quad:
                    file.write(f"f {f[0] + 1}/{f[0] + 1} {f[1] + 1}/{f[1] + 1} {f[2] + 1}/{f[2] + 1} {f[3] + 1}/{f[3] + 1}\n")
            else:
                for f in self.quad:
                    file.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1} {f[3] + 1}\n")







if __name__ == "__main__":
    #### QUAD MESH
    # resolution = 100
    # grid = GridMesh(height=resolution, width=resolution, side_length_h=1, side_length_w=1)
    # save_path = f"../datasets/pgsft/template_grid_{resolution}_debug.obj"
    # grid.save_quad_obj(save_path)
    # print('saved to:', save_path)


    #### TRI MESH
    # resolution = 100
    # grid = GridMesh(height=resolution, width=resolution, side_length=1)
    # # print(grid.mass)
    # save_path = f"../Physics-guided_SfT/data/template_phisft_{resolution}.obj"
    # grid.save_obj(save_path)
    # print('saved to:', save_path)


    ### Non square mesh
    height = 256
    width = 32
    side_length_h = height / 32
    side_length_w = width / 32
    grid = GridMesh(height=height, width=width, side_length_h=side_length_h, side_length_w=side_length_w)
    save_path = f"../datasets/pgsft/template_grid_{height}x{width}.obj"
    grid.save_quad_obj(save_path)
    # grid.save_obj(save_path)
    print('saved to:', save_path)
