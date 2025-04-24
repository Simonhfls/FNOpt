# Ref: https://github.com/jrsassen/repulsive-shells.git
import torch
import torch.nn.functional as F

class TangentPointEnergy:
    def __init__(self,
                 alpha: int = 6,
                 beta: int = 12,
                 **kwargs):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, **kwargs):
        pass

    def _faces_areas_and_normals(self, verts_padded, faces_packed):
        i, j, k = faces_packed.unbind(dim=-1)
        Pi, Pj, Pk = verts_padded[..., i, :], verts_padded[..., j, :], verts_padded[..., k, :]
        normals = torch.cross(Pk - Pj, Pi - Pk, dim=-1)
        dbl_area = torch.linalg.norm(normals, dim=-1)
        return dbl_area / 2., normals / dbl_area.unsqueeze(-1)

    def _faces_barycenters(self, verts_padded, faces_packed):
        i, j, k = faces_packed.unbind(dim=-1)
        Pi, Pj, Pk = verts_padded[..., i, :], verts_padded[..., j, :], verts_padded[..., k, :]
        return (Pi + Pj + Pk) / 3.

    @staticmethod
    def TPEKernel(x, y, n, alpha, beta, eps=1e-12):
        '''
        VecType offset = a - b;
        return my_pow( dotProduct( n, offset ), m_alpha ) / my_pow( offset.norm(), m_beta );
        '''
        d = x - y
        proj = (n * d).sum(dim=-1)
        dist_sq = (d ** 2).sum(dim=-1)

        log_proj = torch.log(torch.abs(proj) + eps) * alpha
        log_dist = torch.log(dist_sq + eps) * (beta / 2.0)
        log_kernel = log_proj - log_dist

        return log_kernel.exp()

    @staticmethod
    def TPEKernel2(x, y, n, alpha, beta, margin=1e-3, eps=1e-12):
        '''
        VecType offset = a - b;
        return my_pow( dotProduct( n, offset ), m_alpha ) / my_pow( offset.norm(), m_beta );
        '''
        d = x - y
        proj = (n * d).sum(dim=-1)
        dist_sq = (d ** 2).sum(dim=-1) + eps
        dist_sq_clamped = torch.clamp(dist_sq, min=margin ** 2)

        log_proj = torch.log(torch.abs(proj) + eps) * alpha
        log_dist = torch.log(dist_sq_clamped) * (beta / 2.0)
        log_kernel = log_proj - log_dist

        return log_kernel.exp()

    def TPEKernel_energy(self, x, y, n, a1, a2, alpha, beta, eps=1e-12, chunk_size=100):
        """
        分块计算能量，避免构造完整的 [N, N] kernel 矩阵。
        参数说明同 __call__ 中，chunk_size 可根据内存情况调整。
        """
        N = x.shape[0]
        energy = 0.0
        for i in range(0, N, chunk_size):
            # 取一部分数据
            x_chunk = x[i:i + chunk_size]  # [chunk_size, 3] 注意：x 原始 shape: [N,3]
            n_chunk = n[i:i + chunk_size]  # [chunk_size, 3]
            a1_chunk = a1[i:i + chunk_size]  # [chunk_size, 1]
            # 构造 pairwise 差分，利用广播，d 的 shape: [chunk_size, N, 3]
            d = x_chunk - y
            # print('i:', i, 'n_chunk:', n_chunk.shape, 'd:', d.shape)
            proj = (n_chunk * d).sum(dim=-1)  # [chunk_size, N]
            dist_sq = (d ** 2).sum(dim=-1)  # [chunk_size, N]
            log_proj = torch.log(torch.abs(proj) + eps) * alpha
            log_dist = torch.log(dist_sq + eps) * (beta / 2.0)
            kernel_chunk = torch.exp(log_proj - log_dist)  # [chunk_size, N]
            # 累计能量，这里 a2 的 shape 为 [1, N]
            energy += (a1_chunk * a2 * kernel_chunk).sum()
        return energy

    def TPEKernel_energy2(self, x, y, n, a1, a2, alpha, beta, eps=1e-12,
                         chunk_size=10):
        """
        分两重循环分块计算能量，避免构造完整的 [N, N] kernel 矩阵，
        从而减少单次内存分配。

        参数：
          x: [N, 3] 顶点坐标（例如重心）
          y: [N, 3] 与 x 比较的另一份顶点数据（通常和 x 相同）
          n: [N, 3] 对应的法向
          a1: [N, 1] 面片面积（对应 x）
          a2: [1, N] 面片面积（对应 y）
          alpha, beta: TPE 参数
          eps: 防止 log 里出现 0 的小正数
          chunk_size_x: 对 x 的分块大小（建议比较小，比如 10）
          chunk_size_y: 对 y 的分块大小（可以比 x 稍大一些，比如 100）
        """
        N = x.shape[0]
        energy = 0.0
        # 分块计算 x 方向
        for i in range(0, N, chunk_size):
            x_chunk = x[i:i + chunk_size]  # [chunk_size, 3]
            n_chunk = n[i:i + chunk_size]  # [chunk_size, 3]
            a1_chunk = a1[i:i + chunk_size]  # [chunk_size, 1]
            # 分块计算 y 方向
            for j in range(0, N, chunk_size):
                y_chunk = y[:, j:j + chunk_size]  # [chunk_size, 3]
                a2_chunk = a2[:, j:j + chunk_size]  # [1, chunk_size]
                # 计算两组之间的 pairwise 差分：结果 [chunk_size, chunk_size, 3]
                d = x_chunk[:, :] - y_chunk[:, :]
                proj = (n_chunk[:, :] * d).sum(dim=-1)  # [chunk_size, chunk_size]
                dist_sq = (d ** 2).sum(dim=-1)  # [chunk_size, chunk_size]
                log_proj = torch.log(torch.abs(proj) + eps) * alpha
                log_dist = torch.log(dist_sq + eps) * (beta / 2.0)
                kernel_chunk = torch.exp(log_proj - log_dist)  # [chunk_size, chunk_size]
                energy += (a1_chunk * a2_chunk * kernel_chunk).sum()
        return energy


class AllPairsTPE(TangentPointEnergy):
    def __call__(self, verts_packed, faces_packed, **kwargs):
        # 1) Precompute face-areas, face-normals, and face-barycenters
        faces_areas_packed, faces_normals_packed = self._faces_areas_and_normals(verts_packed, faces_packed)
        faces_barycenters_packed = self._faces_barycenters(verts_packed, faces_packed)

        # 2) Compute Tangent Point Energy
        c1 = faces_barycenters_packed[..., None, :]
        c2 = faces_barycenters_packed[..., None, :, :]
        n1 = faces_normals_packed[..., None, :]
        a1 = faces_areas_packed[..., None]
        a2 = faces_areas_packed[..., None, :]

        # K = self.TPEKernel(c1, c2, n1, self.alpha, self.beta)
        K = self.TPEKernel2(c1, c2, n1, self.alpha, self.beta)
        energy = (a1 * a2 * K).sum()
        # print('c1:', c1.shape, 'c2:', c2.shape, 'n1:', n1.shape, 'a1:', a1.shape, 'a2:', a2.shape)
        # energy = self.TPEKernel_energy(c1, c2, n1, a1, a2, self.alpha, self.beta, eps=1e-12, chunk_size=100)
        # energy = self.TPEKernel_energy2(c1, c2, n1, a1, a2, self.alpha, self.beta, eps=1e-12, chunk_size=100)

        return energy


class SelfCollisionLoss():
    def __call__(self, points1, points2, margin=0.01):
        # 1) 计算两组点之间的欧几里得距离
        d = (points1 - points2).norm(dim=-1)

        # 2) 当距离小于 margin 时，损失 = (margin - d)^2，否则 0
        #    F.relu(x) = max(0, x)
        penalty = F.relu(margin - d) ** 2

        # penalty 的形状与 d 相同
        return penalty

